#include "cheetah_vulkan.h"
#include "cheetah_color.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHEETAH_DEFAULT_POOL_SIZE ((VkDeviceSize)800 * 1024 * 1024)
#define CHEETAH_POOL_ENV "CHEETAH_VRAM_POOL_SIZE"
#define CHEETAH_POOL_ALIGNMENT ((VkDeviceSize)4096)
#define CHEETAH_SPIRV_MAGIC UINT32_C(0x07230203)
#ifndef CHEETAH_DEFAULT_MATMUL_SPV
#define CHEETAH_DEFAULT_MATMUL_SPV "matmul.spv"
#endif
#ifndef CHEETAH_DEFAULT_MATMUL_BF16_SPV
#define CHEETAH_DEFAULT_MATMUL_BF16_SPV "matmul_bf16.spv"
#endif
#ifndef CHEETAH_DEFAULT_SHARED_OPS_SPV
#define CHEETAH_DEFAULT_SHARED_OPS_SPV "shared_ops.spv"
#endif

typedef struct FreeBlock {
    VkDeviceSize offset;
    VkDeviceSize size;
    struct FreeBlock *next;
} FreeBlock;

typedef struct Allocation {
    void *pointer;
    VkDeviceSize offset;
    VkDeviceSize size;
    FreeBlock *free_reserve;
    struct Allocation *next;
} Allocation;

typedef struct VulkanState {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceProperties properties;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t queue_family_index;
    VkCommandPool command_pool;
    VkBuffer pool_buffer;
    VkDeviceMemory pool_memory;
    void *mapped_memory;
    VkDeviceSize pool_size;
    VkDeviceSize memory_size;
    VkDeviceSize allocation_alignment;
    uint32_t memory_type_index;
    FreeBlock *free_blocks;
    Allocation *allocations;
    int initialized;
} VulkanState;

static VulkanState g_vulkan;
static atomic_flag g_allocator_lock = ATOMIC_FLAG_INIT;
static atomic_flag g_matmul_lock = ATOMIC_FLAG_INIT;
static VkDescriptorSetLayout g_matmul_descriptor_layout = VK_NULL_HANDLE;
static VkPipelineLayout g_matmul_pipeline_layout = VK_NULL_HANDLE;
static VkPipeline g_matmul_pipeline = VK_NULL_HANDLE;
static VkPipeline g_matmul_bf16_pipeline = VK_NULL_HANDLE;
static VkDescriptorSetLayout g_qwen_descriptor_layout = VK_NULL_HANDLE;
static VkPipelineLayout g_qwen_pipeline_layout = VK_NULL_HANDLE;
static VkPipeline g_qwen_pipeline = VK_NULL_HANDLE;

static void destroy_matmul_pipeline(void);
static int matrix_byte_size(int rows, int columns, size_t *bytes_out);

static const char *vk_result_name(VkResult result) {
    switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
#ifdef VK_ERROR_OUT_OF_POOL_MEMORY
    case VK_ERROR_OUT_OF_POOL_MEMORY: return "VK_ERROR_OUT_OF_POOL_MEMORY";
#endif
#ifdef VK_ERROR_INVALID_EXTERNAL_HANDLE
    case VK_ERROR_INVALID_EXTERNAL_HANDLE: return "VK_ERROR_INVALID_EXTERNAL_HANDLE";
#endif
#ifdef VK_ERROR_FRAGMENTATION
    case VK_ERROR_FRAGMENTATION: return "VK_ERROR_FRAGMENTATION";
#endif
#ifdef VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS
    case VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS: return "VK_ERROR_INVALID_OPAQUE_CAPTURE_ADDRESS";
#endif
    default: return "VK_RESULT_UNKNOWN";
    }
}

static int check_vk_result(VkResult result, const char *operation) {
    if (result == VK_SUCCESS) {
        return 0;
    }
    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                    "cheetah-vulkan: %s failed: %s (%d)\n",
                    operation, vk_result_name(result), (int)result);
    return -1;
}

static void report_error(const char *message) {
    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                    "cheetah-vulkan: %s\n", message);
}

static void allocator_lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_allocator_lock,
                                             memory_order_acquire)) {
    }
}

static void allocator_unlock(void) {
    atomic_flag_clear_explicit(&g_allocator_lock, memory_order_release);
}

static void matmul_lock(void) {
    while (atomic_flag_test_and_set_explicit(&g_matmul_lock,
                                             memory_order_acquire)) {
    }
}

static void matmul_unlock(void) {
    atomic_flag_clear_explicit(&g_matmul_lock, memory_order_release);
}

static int parse_pool_size(VkDeviceSize *size_out) {
    const char *value = getenv(CHEETAH_POOL_ENV);
    char *end = NULL;
    unsigned long long parsed;

    if (value == NULL || value[0] == '\0') {
        *size_out = CHEETAH_DEFAULT_POOL_SIZE;
        return 0;
    }
    if (value[0] == '-') {
        report_error(CHEETAH_POOL_ENV " must be a positive byte count");
        return -1;
    }
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        report_error(CHEETAH_POOL_ENV " must be a positive byte count");
        return -1;
    }
    if ((uintmax_t)parsed > (uintmax_t)SIZE_MAX) {
        report_error(CHEETAH_POOL_ENV " exceeds the addressable host size");
        return -1;
    }
    *size_out = (VkDeviceSize)parsed;
    return 0;
}

static int has_required_memory_type(VkPhysicalDevice physical_device,
                                    VkDeviceSize required_size) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint32_t i;

    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for (i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryType *type = &memory_properties.memoryTypes[i];
        const VkMemoryHeap *heap = &memory_properties.memoryHeaps[type->heapIndex];
        const VkMemoryPropertyFlags required =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((type->propertyFlags & required) == required &&
            heap->size >= required_size) {
            return 1;
        }
    }
    return 0;
}

static int select_physical_device(VkDeviceSize pool_size) {
    VkPhysicalDevice *devices = NULL;
    uint32_t device_count = 0;
    uint32_t i;
    VkResult result = vkEnumeratePhysicalDevices(g_vulkan.instance,
                                                  &device_count, NULL);
    if (check_vk_result(result, "vkEnumeratePhysicalDevices(count)") != 0) {
        return -1;
    }
    if (device_count == 0) {
        report_error("no Vulkan physical devices were found");
        return -1;
    }
    devices = (VkPhysicalDevice *)calloc(device_count, sizeof(*devices));
    if (devices == NULL) {
        report_error("unable to allocate physical-device list");
        return -1;
    }
    result = vkEnumeratePhysicalDevices(g_vulkan.instance, &device_count,
                                        devices);
    if (check_vk_result(result, "vkEnumeratePhysicalDevices(list)") != 0) {
        free(devices);
        return -1;
    }

    for (i = 0; i < device_count; ++i) {
        uint32_t family_count = 0;
        VkQueueFamilyProperties *families;
        uint32_t family;

        if (!has_required_memory_type(devices[i], pool_size)) {
            continue;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count,
                                                  NULL);
        if (family_count == 0) {
            continue;
        }
        families = (VkQueueFamilyProperties *)calloc(family_count,
                                                      sizeof(*families));
        if (families == NULL) {
            free(devices);
            report_error("unable to allocate queue-family list");
            return -1;
        }
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count,
                                                  families);
        for (family = 0; family < family_count; ++family) {
            if (families[family].queueCount > 0 &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                g_vulkan.physical_device = devices[i];
                g_vulkan.queue_family_index = family;
                vkGetPhysicalDeviceProperties(devices[i],
                                               &g_vulkan.properties);
                fprintf(stderr,
                        "cheetah-vulkan: device '%s' (API %u.%u.%u, "
                        "discrete=%d)\n",
                        g_vulkan.properties.deviceName,
                        VK_API_VERSION_MAJOR(g_vulkan.properties.apiVersion),
                        VK_API_VERSION_MINOR(g_vulkan.properties.apiVersion),
                        VK_API_VERSION_PATCH(g_vulkan.properties.apiVersion),
                        g_vulkan.properties.deviceType ==
                                VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                            ? 1
                            : 0);
                free(families);
                free(devices);
                return 0;
            }
        }
        free(families);
    }

    free(devices);
    report_error("no Vulkan device has a compute queue and a sufficiently "
                 "large HOST_VISIBLE | HOST_COHERENT memory heap");
    return -1;
}

static int select_pool_memory_type(uint32_t type_bits,
                                   VkDeviceSize required_size,
                                   uint32_t *type_index_out) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    uint32_t i;
    const VkMemoryPropertyFlags required =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    vkGetPhysicalDeviceMemoryProperties(g_vulkan.physical_device,
                                        &memory_properties);
    for (i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const VkMemoryType *type = &memory_properties.memoryTypes[i];
        const VkMemoryHeap *heap = &memory_properties.memoryHeaps[type->heapIndex];
        if ((type_bits & (UINT32_C(1) << i)) != 0 &&
            (type->propertyFlags & required) == required &&
            heap->size >= required_size) {
            *type_index_out = i;
            return 0;
        }
    }
    report_error("the storage buffer requires no compatible memory type with "
                 "HOST_VISIBLE | HOST_COHERENT properties and sufficient heap");
    return -1;
}

static void free_allocator_metadata(void) {
    FreeBlock *block = g_vulkan.free_blocks;
    Allocation *allocation = g_vulkan.allocations;

    while (block != NULL) {
        FreeBlock *next = block->next;
        free(block);
        block = next;
    }
    while (allocation != NULL) {
        Allocation *next = allocation->next;
        free(allocation->free_reserve);
        free(allocation);
        allocation = next;
    }
    g_vulkan.free_blocks = NULL;
    g_vulkan.allocations = NULL;
}

static void destroy_vulkan_state(int wait_for_device) {
    if (g_vulkan.device != VK_NULL_HANDLE) {
        if (wait_for_device) {
            VkResult result = vkDeviceWaitIdle(g_vulkan.device);
            (void)check_vk_result(result, "vkDeviceWaitIdle");
        }
        destroy_matmul_pipeline();
        if (g_vulkan.mapped_memory != NULL) {
            vkUnmapMemory(g_vulkan.device, g_vulkan.pool_memory);
            g_vulkan.mapped_memory = NULL;
        }
        if (g_vulkan.command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(g_vulkan.device, g_vulkan.command_pool, NULL);
            g_vulkan.command_pool = VK_NULL_HANDLE;
        }
        if (g_vulkan.pool_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(g_vulkan.device, g_vulkan.pool_buffer, NULL);
            g_vulkan.pool_buffer = VK_NULL_HANDLE;
        }
        if (g_vulkan.pool_memory != VK_NULL_HANDLE) {
            vkFreeMemory(g_vulkan.device, g_vulkan.pool_memory, NULL);
            g_vulkan.pool_memory = VK_NULL_HANDLE;
        }
        vkDestroyDevice(g_vulkan.device, NULL);
        g_vulkan.device = VK_NULL_HANDLE;
    }
    free_allocator_metadata();
    if (g_vulkan.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(g_vulkan.instance, NULL);
        g_vulkan.instance = VK_NULL_HANDLE;
    }
    memset(&g_vulkan, 0, sizeof(g_vulkan));
    atomic_flag_clear_explicit(&g_allocator_lock, memory_order_release);
    atomic_flag_clear_explicit(&g_matmul_lock, memory_order_release);
}

static VkDeviceSize gcd_device_size(VkDeviceSize a, VkDeviceSize b) {
    while (b != 0) {
        VkDeviceSize remainder = a % b;
        a = b;
        b = remainder;
    }
    return a;
}

int init_vulkan_device(void) {
    VkApplicationInfo application_info;
    VkInstanceCreateInfo instance_info;
    VkDeviceQueueCreateInfo queue_info;
    VkDeviceCreateInfo device_info;
    VkCommandPoolCreateInfo command_pool_info;
    VkBufferCreateInfo buffer_info;
    VkMemoryRequirements memory_requirements;
    VkMemoryAllocateInfo memory_info;
    VkResult result;
    float queue_priority = 1.0f;
    FreeBlock *initial_block;
    VkDeviceSize pool_size;

    if (g_vulkan.initialized) {
        return 0;
    }
    if (g_vulkan.instance != VK_NULL_HANDLE ||
        g_vulkan.device != VK_NULL_HANDLE) {
        destroy_vulkan_state(0);
    }
    if (parse_pool_size(&pool_size) != 0) {
        return -1;
    }
    if (pool_size > (VkDeviceSize)SIZE_MAX) {
        report_error("configured pool size exceeds the addressable host size");
        return -1;
    }
    g_vulkan.pool_size = pool_size;

    memset(&application_info, 0, sizeof(application_info));
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "Cheetah";
    application_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    application_info.pEngineName = "Cheetah";
    application_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    application_info.apiVersion = VK_API_VERSION_1_0;

    memset(&instance_info, 0, sizeof(instance_info));
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &application_info;
    result = vkCreateInstance(&instance_info, NULL, &g_vulkan.instance);
    if (check_vk_result(result, "vkCreateInstance") != 0) {
        destroy_vulkan_state(0);
        return -1;
    }
    if (select_physical_device(pool_size) != 0) {
        destroy_vulkan_state(0);
        return -1;
    }

    memset(&queue_info, 0, sizeof(queue_info));
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = g_vulkan.queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    memset(&device_info, 0, sizeof(device_info));
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    result = vkCreateDevice(g_vulkan.physical_device, &device_info, NULL,
                            &g_vulkan.device);
    if (check_vk_result(result, "vkCreateDevice") != 0) {
        destroy_vulkan_state(0);
        return -1;
    }
    vkGetDeviceQueue(g_vulkan.device, g_vulkan.queue_family_index, 0,
                     &g_vulkan.compute_queue);

    memset(&command_pool_info, 0, sizeof(command_pool_info));
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_info.queueFamilyIndex = g_vulkan.queue_family_index;
    result = vkCreateCommandPool(g_vulkan.device, &command_pool_info, NULL,
                                 &g_vulkan.command_pool);
    if (check_vk_result(result, "vkCreateCommandPool") != 0) {
        destroy_vulkan_state(1);
        return -1;
    }

    memset(&buffer_info, 0, sizeof(buffer_info));
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = pool_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    result = vkCreateBuffer(g_vulkan.device, &buffer_info, NULL,
                            &g_vulkan.pool_buffer);
    if (check_vk_result(result, "vkCreateBuffer(pool)") != 0) {
        destroy_vulkan_state(1);
        return -1;
    }
    vkGetBufferMemoryRequirements(g_vulkan.device, g_vulkan.pool_buffer,
                                  &memory_requirements);
    if (select_pool_memory_type(memory_requirements.memoryTypeBits,
                                memory_requirements.size,
                                &g_vulkan.memory_type_index) != 0) {
        destroy_vulkan_state(1);
        return -1;
    }
    g_vulkan.memory_size = memory_requirements.size;

    memset(&memory_info, 0, sizeof(memory_info));
    memory_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_info.allocationSize = memory_requirements.size;
    memory_info.memoryTypeIndex = g_vulkan.memory_type_index;
    result = vkAllocateMemory(g_vulkan.device, &memory_info, NULL,
                              &g_vulkan.pool_memory);
    if (check_vk_result(result, "vkAllocateMemory(pool)") != 0) {
        destroy_vulkan_state(1);
        return -1;
    }
    result = vkBindBufferMemory(g_vulkan.device, g_vulkan.pool_buffer,
                                g_vulkan.pool_memory, 0);
    if (check_vk_result(result, "vkBindBufferMemory(pool)") != 0) {
        destroy_vulkan_state(1);
        return -1;
    }
    result = vkMapMemory(g_vulkan.device, g_vulkan.pool_memory, 0,
                         VK_WHOLE_SIZE, 0, &g_vulkan.mapped_memory);
    if (check_vk_result(result, "vkMapMemory(pool)") != 0) {
        destroy_vulkan_state(1);
        return -1;
    }

    {
        VkDeviceSize storage_alignment =
            g_vulkan.properties.limits.minStorageBufferOffsetAlignment;
        VkDeviceSize divisor;
        VkDeviceSize factor;
        if (storage_alignment == 0) {
            storage_alignment = 1;
        }
        divisor = gcd_device_size(CHEETAH_POOL_ALIGNMENT, storage_alignment);
        factor = storage_alignment / divisor;
        if (factor > UINT64_MAX / CHEETAH_POOL_ALIGNMENT) {
            report_error("required pool allocation alignment overflows");
            destroy_vulkan_state(1);
            return -1;
        }
        g_vulkan.allocation_alignment = factor * CHEETAH_POOL_ALIGNMENT;
    }
    initial_block = (FreeBlock *)malloc(sizeof(*initial_block));
    if (initial_block == NULL) {
        report_error("unable to allocate initial memory-pool metadata");
        destroy_vulkan_state(1);
        return -1;
    }
    initial_block->offset = 0;
    initial_block->size = pool_size;
    initial_block->next = NULL;
    g_vulkan.free_blocks = initial_block;
    g_vulkan.initialized = 1;

    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_GREEN,
                    "cheetah-vulkan: initialized %s (pool=%" PRIu64
                    " bytes, compute queue family=%u)\n",
                    g_vulkan.properties.deviceName, (uint64_t)pool_size,
                    g_vulkan.queue_family_index);
    return 0;
}

void shutdown_vulkan_device(void) {
    destroy_vulkan_state(1);
}

static int align_up(VkDeviceSize value, VkDeviceSize alignment,
                    VkDeviceSize *aligned_out) {
    VkDeviceSize remainder;
    VkDeviceSize padding;

    if (alignment == 0) {
        return -1;
    }
    remainder = value % alignment;
    padding = remainder == 0 ? 0 : alignment - remainder;
    if (value > UINT64_MAX - padding) {
        return -1;
    }
    *aligned_out = value + padding;
    return 0;
}

void *vk_alloc_vram(size_t size) {
    Allocation *allocation;
    FreeBlock *free_reserve;
    FreeBlock **link;
    VkDeviceSize requested_size = (VkDeviceSize)size;

    if (!g_vulkan.initialized) {
        report_error("vk_alloc_vram called before successful initialization");
        return NULL;
    }
    if (size == 0 || (size_t)requested_size != size) {
        report_error("allocation size must be positive and representable");
        return NULL;
    }
    if (requested_size > g_vulkan.properties.limits.maxStorageBufferRange) {
        report_error("allocation exceeds maxStorageBufferRange");
        return NULL;
    }
    allocation = (Allocation *)malloc(sizeof(*allocation));
    free_reserve = (FreeBlock *)malloc(sizeof(*free_reserve));
    if (allocation == NULL || free_reserve == NULL) {
        free(allocation);
        free(free_reserve);
        report_error("unable to allocate memory-pool bookkeeping");
        return NULL;
    }

    allocator_lock();
    link = &g_vulkan.free_blocks;
    while (*link != NULL) {
        FreeBlock *block = *link;
        VkDeviceSize aligned_offset;
        VkDeviceSize prefix;
        VkDeviceSize suffix;

        if (align_up(block->offset, g_vulkan.allocation_alignment,
                     &aligned_offset) != 0 ||
            aligned_offset < block->offset) {
            link = &block->next;
            continue;
        }
        prefix = aligned_offset - block->offset;
        if (prefix > block->size ||
            requested_size > block->size - prefix) {
            link = &block->next;
            continue;
        }
        suffix = block->size - prefix - requested_size;
        if (prefix != 0 && suffix != 0) {
            FreeBlock *tail = (FreeBlock *)malloc(sizeof(*tail));
            if (tail == NULL) {
                allocator_unlock();
                free(allocation);
                free(free_reserve);
                report_error("unable to split free memory-pool block");
                return NULL;
            }
            tail->offset = aligned_offset + requested_size;
            tail->size = suffix;
            tail->next = block->next;
            block->size = prefix;
            block->next = tail;
        } else if (prefix != 0) {
            block->size = prefix;
        } else if (suffix != 0) {
            block->offset = aligned_offset + requested_size;
            block->size = suffix;
        } else {
            *link = block->next;
            free(block);
        }

        allocation->offset = aligned_offset;
        allocation->size = requested_size;
        allocation->free_reserve = free_reserve;
        allocation->pointer =
            (unsigned char *)g_vulkan.mapped_memory + (size_t)aligned_offset;
        allocation->next = g_vulkan.allocations;
        g_vulkan.allocations = allocation;
        allocator_unlock();
        return allocation->pointer;
    }
    allocator_unlock();
    free(allocation);
    free(free_reserve);
    report_error("Vulkan memory pool is exhausted or fragmented");
    return NULL;
}

static void insert_free_block(FreeBlock *block, VkDeviceSize offset,
                              VkDeviceSize size) {
    FreeBlock **link = &g_vulkan.free_blocks;

    while (*link != NULL && (*link)->offset < offset) {
        link = &(*link)->next;
    }
    block->offset = offset;
    block->size = size;
    block->next = *link;
    *link = block;

    if (block->next != NULL &&
        block->offset + block->size == block->next->offset) {
        FreeBlock *next = block->next;
        block->size += next->size;
        block->next = next->next;
        free(next);
    }
    if (link != &g_vulkan.free_blocks) {
        FreeBlock *previous = g_vulkan.free_blocks;
        while (previous->next != block) {
            previous = previous->next;
        }
        if (previous->offset + previous->size == block->offset) {
            previous->size += block->size;
            previous->next = block->next;
            free(block);
        }
    }
}

void vk_free_vram(void *pointer) {
    Allocation **link;
    Allocation *allocation;

    if (pointer == NULL) {
        return;
    }
    if (!g_vulkan.initialized) {
        report_error("vk_free_vram called before successful initialization");
        return;
    }
    allocator_lock();
    link = &g_vulkan.allocations;
    while (*link != NULL && (*link)->pointer != pointer) {
        link = &(*link)->next;
    }
    if (*link == NULL) {
        allocator_unlock();
        report_error("vk_free_vram received an unknown or already-freed pointer");
        return;
    }
    allocation = *link;
    *link = allocation->next;
    insert_free_block(allocation->free_reserve, allocation->offset,
                      allocation->size);
    allocator_unlock();
    free(allocation);
}

int vk_get_vram_allocation(void *pointer, VkBuffer *buffer,
                           VkDeviceSize *offset, VkDeviceSize *size) {
    Allocation *allocation;

    if (pointer == NULL || buffer == NULL || offset == NULL || size == NULL) {
        report_error("vk_get_vram_allocation requires non-null arguments");
        return -1;
    }
    if (!g_vulkan.initialized) {
        report_error("vk_get_vram_allocation called before initialization");
        return -1;
    }
    allocator_lock();
    for (allocation = g_vulkan.allocations; allocation != NULL;
         allocation = allocation->next) {
        if (allocation->pointer == pointer) {
            *buffer = g_vulkan.pool_buffer;
            *offset = allocation->offset;
            *size = allocation->size;
            allocator_unlock();
            return 0;
        }
    }
    allocator_unlock();
    report_error("allocation pointer is unknown or has already been freed");
    return -1;
}

static uint32_t byte_swap_u32(uint32_t value) {
    return ((value & UINT32_C(0x000000ff)) << 24) |
           ((value & UINT32_C(0x0000ff00)) << 8) |
           ((value & UINT32_C(0x00ff0000)) >> 8) |
           ((value & UINT32_C(0xff000000)) >> 24);
}

VkShaderModule vk_load_spirv_shader(const char *path) {
    FILE *file;
    long file_size;
    size_t byte_count;
    uint32_t *code;
    uint32_t magic;
    size_t i;
    VkShaderModuleCreateInfo shader_info;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result;

    if (!g_vulkan.initialized || path == NULL || path[0] == '\0') {
        report_error("shader loading requires an initialized device and path");
        return VK_NULL_HANDLE;
    }
    file = fopen(path, "rb");
    if (file == NULL) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: cannot open SPIR-V file '%s': %s\n",
                        path, strerror(errno));
        return VK_NULL_HANDLE;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: cannot determine SPIR-V file size '%s'\n",
                        path);
        fclose(file);
        return VK_NULL_HANDLE;
    }
    if (file_size < 4 || (file_size % 4) != 0 ||
        (uintmax_t)file_size > (uintmax_t)SIZE_MAX) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: invalid SPIR-V file size in '%s'\n",
                        path);
        fclose(file);
        return VK_NULL_HANDLE;
    }
    byte_count = (size_t)file_size;
    code = (uint32_t *)malloc(byte_count);
    if (code == NULL) {
        report_error("unable to allocate SPIR-V file buffer");
        fclose(file);
        return VK_NULL_HANDLE;
    }
    if (fread(code, 1, byte_count, file) != byte_count) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: failed to read SPIR-V file '%s'\n",
                        path);
        free(code);
        fclose(file);
        return VK_NULL_HANDLE;
    }
    fclose(file);

    magic = code[0];
    if (magic == byte_swap_u32(CHEETAH_SPIRV_MAGIC)) {
        for (i = 0; i < byte_count / sizeof(*code); ++i) {
            code[i] = byte_swap_u32(code[i]);
        }
        magic = code[0];
    }
    if (magic != CHEETAH_SPIRV_MAGIC) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: invalid SPIR-V magic in '%s'\n", path);
        free(code);
        return VK_NULL_HANDLE;
    }

    memset(&shader_info, 0, sizeof(shader_info));
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = byte_count;
    shader_info.pCode = code;
    result = vkCreateShaderModule(g_vulkan.device, &shader_info, NULL, &shader);
    free(code);
    if (check_vk_result(result, "vkCreateShaderModule") != 0) {
        return VK_NULL_HANDLE;
    }
    return shader;
}

VkPipeline vk_create_compute_pipeline(VkShaderModule shader,
                                      VkPipelineLayout layout,
                                      const char *entry_point) {
    VkComputePipelineCreateInfo pipeline_info;
    VkPipelineShaderStageCreateInfo stage_info;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkResult result;

    if (!g_vulkan.initialized || shader == VK_NULL_HANDLE ||
        layout == VK_NULL_HANDLE || entry_point == NULL ||
        entry_point[0] == '\0') {
        report_error("compute pipeline requires initialized device, shader, "
                     "layout, and entry point");
        return VK_NULL_HANDLE;
    }
    memset(&stage_info, 0, sizeof(stage_info));
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader;
    stage_info.pName = entry_point;

    memset(&pipeline_info, 0, sizeof(pipeline_info));
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = layout;
    pipeline_info.basePipelineIndex = -1;
    result = vkCreateComputePipelines(g_vulkan.device, VK_NULL_HANDLE, 1,
                                      &pipeline_info, NULL, &pipeline);
    if (check_vk_result(result, "vkCreateComputePipelines") != 0) {
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

static void destroy_matmul_pipeline(void) {
    if (g_vulkan.device != VK_NULL_HANDLE) {
        if (g_matmul_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(g_vulkan.device, g_matmul_pipeline, NULL);
        }
        if (g_matmul_bf16_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(g_vulkan.device, g_matmul_bf16_pipeline, NULL);
        }
        if (g_qwen_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(g_vulkan.device, g_qwen_pipeline, NULL);
        }
        if (g_matmul_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(g_vulkan.device,
                                    g_matmul_pipeline_layout, NULL);
        }
        if (g_matmul_descriptor_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(g_vulkan.device,
                                         g_matmul_descriptor_layout, NULL);
        }
        if (g_qwen_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(g_vulkan.device, g_qwen_pipeline_layout,
                                    NULL);
        }
        if (g_qwen_descriptor_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(g_vulkan.device,
                                         g_qwen_descriptor_layout, NULL);
        }
    }
    g_matmul_pipeline = VK_NULL_HANDLE;
    g_matmul_bf16_pipeline = VK_NULL_HANDLE;
    g_matmul_pipeline_layout = VK_NULL_HANDLE;
    g_matmul_descriptor_layout = VK_NULL_HANDLE;
    g_qwen_pipeline = VK_NULL_HANDLE;
    g_qwen_pipeline_layout = VK_NULL_HANDLE;
    g_qwen_descriptor_layout = VK_NULL_HANDLE;
}

static int ensure_matmul_pipeline(int bfloat16_weights) {
    VkDescriptorSetLayoutBinding bindings[3];
    VkDescriptorSetLayoutCreateInfo descriptor_layout_info;
    VkPushConstantRange push_constant;
    VkPipelineLayoutCreateInfo pipeline_layout_info;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result;
    VkPipeline *pipeline = bfloat16_weights ? &g_matmul_bf16_pipeline
                                            : &g_matmul_pipeline;
    const char *shader_path = getenv(bfloat16_weights
                                         ? "CHEETAH_MATMUL_BF16_SPV"
                                         : "CHEETAH_MATMUL_SPV");

    if (*pipeline != VK_NULL_HANDLE) {
        return 0;
    }
    if (shader_path == NULL || shader_path[0] == '\0') {
        shader_path = bfloat16_weights ? CHEETAH_DEFAULT_MATMUL_BF16_SPV
                                       : CHEETAH_DEFAULT_MATMUL_SPV;
    }
    if (g_matmul_descriptor_layout == VK_NULL_HANDLE) {
        memset(bindings, 0, sizeof(bindings));
        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        memset(&descriptor_layout_info, 0, sizeof(descriptor_layout_info));
        descriptor_layout_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_layout_info.bindingCount = 3;
        descriptor_layout_info.pBindings = bindings;
        result = vkCreateDescriptorSetLayout(g_vulkan.device,
                                              &descriptor_layout_info, NULL,
                                              &g_matmul_descriptor_layout);
        if (check_vk_result(result, "vkCreateDescriptorSetLayout(matmul)") != 0) {
            return -1;
        }
    }
    if (g_matmul_pipeline_layout == VK_NULL_HANDLE) {
        memset(&push_constant, 0, sizeof(push_constant));
        push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant.offset = 0;
            push_constant.size = 4 * sizeof(uint32_t);
        memset(&pipeline_layout_info, 0, sizeof(pipeline_layout_info));
        pipeline_layout_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &g_matmul_descriptor_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_constant;
        result = vkCreatePipelineLayout(g_vulkan.device, &pipeline_layout_info,
                                        NULL, &g_matmul_pipeline_layout);
        if (check_vk_result(result, "vkCreatePipelineLayout(matmul)") != 0) {
            return -1;
        }
    }
    shader = vk_load_spirv_shader(shader_path);
    if (shader == VK_NULL_HANDLE) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: unable to load matmul shader '%s'\n",
                        shader_path);
        return -1;
    }
    *pipeline = vk_create_compute_pipeline(shader, g_matmul_pipeline_layout,
                                          "main");
    vkDestroyShaderModule(g_vulkan.device, shader, NULL);
    if (*pipeline == VK_NULL_HANDLE) {
        return -1;
    }
    return 0;
}

static int ensure_qwen_pipeline(void) {
    VkDescriptorSetLayoutBinding bindings[5];
    VkDescriptorSetLayoutCreateInfo descriptor_info;
    VkPushConstantRange push_constant;
    VkPipelineLayoutCreateInfo pipeline_info;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkResult result;
    const char *shader_path = getenv("CHEETAH_QWEN_OPS_SPV");

    if (g_qwen_pipeline != VK_NULL_HANDLE) return 0;
    if (shader_path == NULL || shader_path[0] == '\0') {
        shader_path = CHEETAH_DEFAULT_SHARED_OPS_SPV;
    }
    if (g_qwen_descriptor_layout == VK_NULL_HANDLE) {
        memset(bindings, 0, sizeof(bindings));
        for (uint32_t i = 0; i < 5; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        memset(&descriptor_info, 0, sizeof(descriptor_info));
        descriptor_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_info.bindingCount = 5;
        descriptor_info.pBindings = bindings;
        result = vkCreateDescriptorSetLayout(g_vulkan.device, &descriptor_info,
                                              NULL, &g_qwen_descriptor_layout);
        if (check_vk_result(result, "vkCreateDescriptorSetLayout(qwen)") != 0)
            return -1;
    }
    if (g_qwen_pipeline_layout == VK_NULL_HANDLE) {
        memset(&push_constant, 0, sizeof(push_constant));
        push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant.size = 8 * sizeof(uint32_t);
        memset(&pipeline_info, 0, sizeof(pipeline_info));
        pipeline_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_info.setLayoutCount = 1;
        pipeline_info.pSetLayouts = &g_qwen_descriptor_layout;
        pipeline_info.pushConstantRangeCount = 1;
        pipeline_info.pPushConstantRanges = &push_constant;
        result = vkCreatePipelineLayout(g_vulkan.device, &pipeline_info, NULL,
                                        &g_qwen_pipeline_layout);
        if (check_vk_result(result, "vkCreatePipelineLayout(qwen)") != 0)
            return -1;
    }
    shader = vk_load_spirv_shader(shader_path);
    if (shader == VK_NULL_HANDLE) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-vulkan: unable to load Qwen ops shader '%s'\n",
                        shader_path);
        return -1;
    }
    g_qwen_pipeline =
        vk_create_compute_pipeline(shader, g_qwen_pipeline_layout, "main");
    vkDestroyShaderModule(g_vulkan.device, shader, NULL);
    return g_qwen_pipeline == VK_NULL_HANDLE ? -1 : 0;
}

int vk_qwen_prepare(void) {
    int status;
    if (!g_vulkan.initialized) {
        report_error("Vulkan device is not initialized");
        return -1;
    }
    matmul_lock();
    status = ensure_qwen_pipeline();
    matmul_unlock();
    return status;
}

static int run_qwen_compute(const void *const sources[5],
                            void *const destinations[5],
                            const size_t byte_sizes[5],
                            const uint32_t parameters[8],
                            uint32_t group_count_x) {
    void *device_buffers[5] = {NULL, NULL, NULL, NULL, NULL};
    VkBuffer buffers[5];
    VkDeviceSize offsets[5];
    VkDeviceSize ranges[5];
    VkDescriptorBufferInfo buffer_infos[5];
    VkWriteDescriptorSet writes[5];
    VkDescriptorPoolSize pool_size;
    VkDescriptorPoolCreateInfo pool_info;
    VkDescriptorSetAllocateInfo set_info;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo command_info;
    VkCommandBufferBeginInfo begin_info;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFenceCreateInfo fence_info;
    VkFence fence = VK_NULL_HANDLE;
    VkSubmitInfo submit_info;
    VkResult result;
    int command_allocated = 0;
    int submitted = 0;
    int completed = 0;
    int status = -1;
    uint32_t i;

    if (!g_vulkan.initialized || parameters == NULL || group_count_x == 0) {
        report_error("invalid Qwen compute dispatch");
        return -1;
    }
    if (group_count_x > g_vulkan.properties.limits.maxComputeWorkGroupCount[0]) {
        report_error("Qwen dispatch exceeds maxComputeWorkGroupCount");
        return -1;
    }
    matmul_lock();
    if (ensure_qwen_pipeline() != 0) goto cleanup;
    for (i = 0; i < 5; ++i) {
        size_t allocation_size;
        if (byte_sizes[i] > SIZE_MAX - 3) {
            report_error("Qwen buffer size overflows size_t");
            goto cleanup;
        }
        allocation_size = (byte_sizes[i] + 3U) & ~(size_t)3U;
        if (allocation_size == 0) allocation_size = 4;
        if ((VkDeviceSize)allocation_size >
            g_vulkan.properties.limits.maxStorageBufferRange) {
            report_error("Qwen tensor exceeds maxStorageBufferRange");
            goto cleanup;
        }
        device_buffers[i] = vk_alloc_vram(allocation_size);
        if (device_buffers[i] == NULL) goto cleanup;
        if (sources[i] != NULL && byte_sizes[i] > 0) {
            memcpy(device_buffers[i], sources[i], byte_sizes[i]);
        } else {
            memset(device_buffers[i], 0, allocation_size);
        }
        if (destinations[i] != NULL) {
            memset(device_buffers[i], 0, allocation_size);
        }
        if (vk_get_vram_allocation(device_buffers[i], &buffers[i],
                                   &offsets[i], &ranges[i]) != 0) {
            goto cleanup;
        }
        buffer_infos[i].buffer = buffers[i];
        buffer_infos[i].offset = offsets[i];
        buffer_infos[i].range = ranges[i];
    }
    memset(&pool_size, 0, sizeof(pool_size));
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = 5;
    memset(&pool_info, 0, sizeof(pool_info));
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    result = vkCreateDescriptorPool(g_vulkan.device, &pool_info, NULL,
                                    &descriptor_pool);
    if (check_vk_result(result, "vkCreateDescriptorPool(qwen)") != 0)
        goto cleanup;
    memset(&set_info, 0, sizeof(set_info));
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_info.descriptorPool = descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &g_qwen_descriptor_layout;
    result = vkAllocateDescriptorSets(g_vulkan.device, &set_info,
                                      &descriptor_set);
    if (check_vk_result(result, "vkAllocateDescriptorSets(qwen)") != 0)
        goto cleanup;
    memset(writes, 0, sizeof(writes));
    for (i = 0; i < 5; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descriptor_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(g_vulkan.device, 5, writes, 0, NULL);

    memset(&command_info, 0, sizeof(command_info));
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_info.commandPool = g_vulkan.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(g_vulkan.device, &command_info,
                                      &command_buffer);
    if (check_vk_result(result, "vkAllocateCommandBuffers(qwen)") != 0)
        goto cleanup;
    command_allocated = 1;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command_buffer, &begin_info);
    if (check_vk_result(result, "vkBeginCommandBuffer(qwen)") != 0)
        goto cleanup;
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      g_qwen_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_qwen_pipeline_layout, 0, 1, &descriptor_set,
                            0, NULL);
    vkCmdPushConstants(command_buffer, g_qwen_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       8 * sizeof(uint32_t), parameters);
    vkCmdDispatch(command_buffer, group_count_x, 1, 1);
    result = vkEndCommandBuffer(command_buffer);
    if (check_vk_result(result, "vkEndCommandBuffer(qwen)") != 0)
        goto cleanup;
    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(g_vulkan.device, &fence_info, NULL, &fence);
    if (check_vk_result(result, "vkCreateFence(qwen)") != 0) goto cleanup;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    result = vkQueueSubmit(g_vulkan.compute_queue, 1, &submit_info, fence);
    if (check_vk_result(result, "vkQueueSubmit(qwen)") != 0) goto cleanup;
    submitted = 1;
    result = vkWaitForFences(g_vulkan.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (check_vk_result(result, "vkWaitForFences(qwen)") != 0) goto cleanup;
    completed = 1;
    for (i = 0; i < 5; ++i) {
        if (destinations[i] != NULL && byte_sizes[i] > 0) {
            memcpy(destinations[i], device_buffers[i], byte_sizes[i]);
        }
    }
    status = 0;

cleanup:
    if (fence != VK_NULL_HANDLE) vkDestroyFence(g_vulkan.device, fence, NULL);
    if (command_allocated) {
        if (submitted && !completed) {
            VkResult idle_result = vkDeviceWaitIdle(g_vulkan.device);
            (void)check_vk_result(idle_result, "vkDeviceWaitIdle(qwen cleanup)");
        }
        vkFreeCommandBuffers(g_vulkan.device, g_vulkan.command_pool, 1,
                             &command_buffer);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(g_vulkan.device, descriptor_pool, NULL);
    }
    for (i = 0; i < 5; ++i) {
        if (device_buffers[i] != NULL) vk_free_vram(device_buffers[i]);
    }
    matmul_unlock();
    return status;
}

static uint32_t ceil_div_u32(uint32_t numerator, uint32_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
}

static int tensor_element_count(uint32_t a, uint32_t b, uint32_t c,
                               uint64_t *count_out) {
    uint64_t count = a;
    if (b != 0 && count > UINT64_MAX / b) return -1;
    count *= b;
    if (c != 0 && count > UINT64_MAX / c) return -1;
    *count_out = count * c;
    return 0;
}

int vk_qwen_rms_norm(const float *input, const void *weight,
                     uint32_t weight_format, float *output, uint32_t rows,
                     uint32_t width, float epsilon) {
    const void *sources[5] = {input, weight, NULL, NULL, NULL};
    void *destinations[5] = {NULL, NULL, output, NULL, NULL};
    size_t sizes[5];
    uint32_t parameters[8] = {0};
    size_t input_bytes;
    size_t weight_element_size;
    size_t weight_bytes;
    if (input == NULL || weight == NULL || output == NULL ||
        rows == 0 || width == 0 || rows > INT_MAX || width > INT_MAX ||
        weight_format > 2 || !isfinite(epsilon) || epsilon < 0.0f ||
        matrix_byte_size((int)rows, (int)width, &input_bytes) != 0) {
        report_error("invalid arguments to vk_qwen_rms_norm");
        return -1;
    }
    weight_element_size = weight_format == 0 ? sizeof(float) : sizeof(uint16_t);
    if ((size_t)width > SIZE_MAX / weight_element_size) {
        report_error("Qwen normalization weight size overflows size_t");
        return -1;
    }
    weight_bytes = (size_t)width * weight_element_size;
    sizes[0] = input_bytes;
    sizes[1] = weight_bytes;
    sizes[2] = input_bytes;
    sizes[3] = 0;
    sizes[4] = 0;
    parameters[0] = 0;
    parameters[1] = rows;
    parameters[2] = width;
    memcpy(&parameters[3], &epsilon, sizeof(epsilon));
    parameters[4] = weight_format;
    return run_qwen_compute(sources, destinations, sizes, parameters,
                            ceil_div_u32(rows, 64));
}

int vk_qwen_layer_norm(const float *input, const void *weight,
                       const void *bias, uint32_t weight_format,
                       float *output, uint32_t rows, uint32_t width,
                       float epsilon) {
    const void *sources[5] = {input, weight, NULL, NULL, bias};
    void *destinations[5] = {NULL, NULL, output, NULL, NULL};
    size_t sizes[5];
    uint32_t parameters[8] = {0};
    size_t input_bytes;
    size_t weight_element_size;
    size_t weight_bytes;
    if (input == NULL || weight == NULL || bias == NULL || output == NULL ||
        rows == 0 || width == 0 || rows > INT_MAX || width > INT_MAX ||
        weight_format > 2 || !isfinite(epsilon) || epsilon < 0.0f ||
        matrix_byte_size((int)rows, (int)width, &input_bytes) != 0) {
        report_error("invalid arguments to vk_qwen_layer_norm");
        return -1;
    }
    weight_element_size = weight_format == 0 ? sizeof(float) : sizeof(uint16_t);
    if ((size_t)width > SIZE_MAX / weight_element_size) {
        report_error("layer normalization weight size overflows size_t");
        return -1;
    }
    weight_bytes = (size_t)width * weight_element_size;
    sizes[0] = input_bytes;
    sizes[1] = weight_bytes;
    sizes[2] = input_bytes;
    sizes[3] = 0;
    sizes[4] = weight_bytes;
    parameters[0] = 7;
    parameters[1] = rows;
    parameters[2] = width;
    memcpy(&parameters[3], &epsilon, sizeof(epsilon));
    parameters[4] = weight_format;
    return run_qwen_compute(sources, destinations, sizes, parameters,
                            ceil_div_u32(rows, 64));
}

int vk_qwen_elementwise(const float *a, const float *b, float *output,
                        size_t count, uint32_t operation) {
    const void *sources[5] = {a, b, NULL, NULL, NULL};
    void *destinations[5] = {NULL, NULL, output, NULL, NULL};
    size_t sizes[5];
    uint32_t parameters[8] = {0};
    size_t bytes;
    if (a == NULL || b == NULL || output == NULL || count == 0 ||
        (operation != 1 && operation != 2 &&
         operation != 5 && operation != 6) ||
        count > SIZE_MAX / sizeof(float)) {
        report_error("invalid arguments to vk_qwen_elementwise");
        return -1;
    }
    bytes = count * sizeof(float);
    sizes[0] = sizes[1] = sizes[2] = bytes;
    sizes[3] = sizes[4] = 0;
    parameters[0] = operation;
    if (count > UINT32_MAX) {
        report_error("Qwen elementwise count exceeds 32-bit shader indexing");
        return -1;
    }
    parameters[1] = (uint32_t)count;
    return run_qwen_compute(sources, destinations, sizes, parameters,
                            ceil_div_u32((uint32_t)count, 64));
}

int vk_qwen_rotary(const float *query, const float *keys,
                   float *query_output, float *key_output,
                   uint32_t sequence, uint32_t offset,
                   uint32_t query_heads, uint32_t kv_heads,
                   uint32_t head_dim, uint32_t rotary_dim, float theta) {
    const void *sources[5] = {query, keys, NULL, NULL, NULL};
    void *destinations[5] = {NULL, NULL, query_output, key_output, NULL};
    size_t sizes[5];
    uint32_t parameters[8] = {0};
    uint64_t query_count;
    uint64_t key_count;
    if (query == NULL || keys == NULL || query_output == NULL ||
        key_output == NULL || sequence == 0 || query_heads == 0 ||
        kv_heads == 0 || head_dim == 0 || rotary_dim == 0 ||
        rotary_dim > head_dim ||
        query_heads % kv_heads != 0 || (head_dim & 1U) != 0 ||
        (rotary_dim & 1U) != 0 ||
        !isfinite(theta) || theta <= 0.0f) {
        report_error("invalid arguments to vk_qwen_rotary");
        return -1;
    }
    if (tensor_element_count(sequence, query_heads, head_dim, &query_count) != 0 ||
        tensor_element_count(sequence, kv_heads, head_dim, &key_count) != 0 ||
        query_count > SIZE_MAX / sizeof(float) ||
        key_count > SIZE_MAX / sizeof(float) ||
        query_count > UINT32_MAX ||
        key_count > UINT32_MAX - query_count) {
        report_error("Qwen rotary tensor exceeds supported shader indexing");
        return -1;
    }
    sizes[0] = sizes[2] = (size_t)query_count * sizeof(float);
    sizes[1] = sizes[3] = (size_t)key_count * sizeof(float);
    sizes[4] = 0;
    parameters[0] = 3;
    parameters[1] = sequence;
    parameters[2] = offset;
    parameters[3] = query_heads;
    parameters[4] = kv_heads;
    parameters[5] = head_dim;
    memcpy(&parameters[6], &theta, sizeof(theta));
    parameters[7] = rotary_dim;
    return run_qwen_compute(sources, destinations, sizes, parameters,
                            ceil_div_u32((uint32_t)(query_count + key_count), 64));
}

int vk_qwen_attention(const float *query, const float *keys,
                      const float *values, float *output,
                      uint32_t sequence, uint32_t offset,
                      uint32_t cache_len, uint32_t query_heads,
                      uint32_t kv_heads, uint32_t head_dim,
                      uint32_t sliding_window) {
    const void *sources[5] = {query, keys, NULL, NULL, values};
    void *destinations[5] = {NULL, NULL, NULL, output, NULL};
    size_t sizes[5];
    uint32_t parameters[8] = {0};
    uint64_t query_count;
    uint64_t kv_count;
    if (query == NULL || keys == NULL || values == NULL || output == NULL ||
        sequence == 0 || query_heads == 0 || kv_heads == 0 || head_dim == 0 ||
        query_heads % kv_heads != 0 || offset > UINT32_MAX - sequence ||
        cache_len < offset + sequence) {
        report_error("invalid arguments to vk_qwen_attention");
        return -1;
    }
    if (tensor_element_count(sequence, query_heads, head_dim, &query_count) != 0 ||
        tensor_element_count(cache_len, kv_heads, head_dim, &kv_count) != 0 ||
        query_count > UINT32_MAX || kv_count > UINT32_MAX ||
        query_count > SIZE_MAX / sizeof(float) ||
        kv_count > SIZE_MAX / sizeof(float) ||
        (uint64_t)sequence * query_heads > UINT32_MAX) {
        report_error("Qwen attention tensor exceeds supported shader indexing");
        return -1;
    }
    sizes[0] = sizes[3] = (size_t)query_count * sizeof(float);
    sizes[1] = sizes[4] = (size_t)kv_count * sizeof(float);
    sizes[2] = 0;
    parameters[0] = 4;
    parameters[1] = sequence;
    parameters[2] = offset;
    parameters[3] = cache_len;
    parameters[4] = query_heads;
    parameters[5] = kv_heads;
    parameters[6] = head_dim;
    parameters[7] = sliding_window;
    return run_qwen_compute(sources, destinations, sizes, parameters,
                            ceil_div_u32(sequence * query_heads, 64));
}

static int matrix_byte_size(int rows, int columns, size_t *bytes_out) {
    size_t row_count = (size_t)rows;
    size_t column_count = (size_t)columns;

    if (row_count > SIZE_MAX / column_count) {
        report_error("matrix element count overflows size_t");
        return -1;
    }
    row_count *= column_count;
    if (row_count > SIZE_MAX / sizeof(float)) {
        report_error("matrix byte size overflows size_t");
        return -1;
    }
    *bytes_out = row_count * sizeof(float);
    return 0;
}

static int vk_matmul_impl(const float *a, const void *b, float *c,
                          int m, int k, int n, int weight_format,
                          int transpose_b) {
    void *device_a = NULL;
    void *device_b = NULL;
    void *device_c = NULL;
    VkBuffer buffer_a = VK_NULL_HANDLE;
    VkBuffer buffer_b = VK_NULL_HANDLE;
    VkBuffer buffer_c = VK_NULL_HANDLE;
    VkDeviceSize offset_a = 0;
    VkDeviceSize offset_b = 0;
    VkDeviceSize offset_c = 0;
    VkDeviceSize size_a = 0;
    VkDeviceSize size_b = 0;
    VkDeviceSize size_c = 0;
    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDescriptorPoolSize descriptor_pool_size;
    VkDescriptorPoolCreateInfo descriptor_pool_info;
    VkDescriptorSetAllocateInfo descriptor_allocate_info;
    VkDescriptorBufferInfo buffer_infos[3];
    VkWriteDescriptorSet descriptor_writes[3];
    VkCommandBufferAllocateInfo command_allocate_info;
    VkCommandBufferBeginInfo command_begin_info;
    VkFenceCreateInfo fence_info;
    VkSubmitInfo submit_info;
    uint32_t dimensions[4];
    uint32_t groups_x;
    uint32_t groups_y;
    size_t bytes_a;
    size_t bytes_b;
    size_t bytes_c;
    int command_allocated = 0;
    int submitted = 0;
    int wait_succeeded = 0;
    int status = -1;
    VkResult result;

    if (a == NULL || b == NULL || c == NULL || m <= 0 || k <= 0 || n <= 0) {
        report_error("vk_matmul requires non-null matrices and positive dimensions");
        return -1;
    }
    if (!g_vulkan.initialized) {
        report_error("vk_matmul called before successful initialization");
        return -1;
    }
    if (matrix_byte_size(m, k, &bytes_a) != 0 ||
        matrix_byte_size(k, n, &bytes_b) != 0 ||
        matrix_byte_size(m, n, &bytes_c) != 0) {
        return -1;
    }
    if (weight_format != 0) {
        bytes_b /= 2;
        if ((bytes_b & 3U) != 0) {
            if (bytes_b > SIZE_MAX - 2) {
                report_error("BF16 weight buffer size overflows size_t");
                return -1;
            }
            bytes_b += 2;
        }
    }
    if (bytes_a / sizeof(float) > UINT32_MAX ||
        (weight_format != 0 ? bytes_b / sizeof(uint16_t)
                            : bytes_b / sizeof(float)) > UINT32_MAX ||
        bytes_c / sizeof(float) > UINT32_MAX) {
        report_error("matrix element count exceeds shader 32-bit indexing");
        return -1;
    }
    if (g_vulkan.properties.limits.maxComputeWorkGroupInvocations < 256 ||
        g_vulkan.properties.limits.maxComputeWorkGroupSize[0] < 16 ||
        g_vulkan.properties.limits.maxComputeWorkGroupSize[1] < 16) {
        report_error("device does not support the matmul shader's 16x16 workgroup");
        return -1;
    }
    groups_x = ((uint32_t)n + 15U) / 16U;
    groups_y = ((uint32_t)m + 15U) / 16U;
    if (groups_x > g_vulkan.properties.limits.maxComputeWorkGroupCount[0] ||
        groups_y > g_vulkan.properties.limits.maxComputeWorkGroupCount[1]) {
        report_error("matrix dimensions exceed device compute dispatch limits");
        return -1;
    }
    if ((VkDeviceSize)bytes_a >
            g_vulkan.properties.limits.maxStorageBufferRange ||
        (VkDeviceSize)bytes_b >
            g_vulkan.properties.limits.maxStorageBufferRange ||
        (VkDeviceSize)bytes_c >
            g_vulkan.properties.limits.maxStorageBufferRange) {
        report_error("matrix allocation exceeds maxStorageBufferRange");
        return -1;
    }

    matmul_lock();
    if (ensure_matmul_pipeline(weight_format != 0) != 0) {
        goto cleanup;
    }
    device_a = vk_alloc_vram(bytes_a);
    device_b = vk_alloc_vram(bytes_b);
    device_c = vk_alloc_vram(bytes_c);
    if (device_a == NULL || device_b == NULL || device_c == NULL) {
        goto cleanup;
    }
    memcpy(device_a, a, bytes_a);
    memset(device_b, 0, bytes_b);
    memcpy(device_b, b, weight_format != 0
                            ? (size_t)k * (size_t)n * sizeof(uint16_t)
                            : bytes_b);
    memset(device_c, 0, bytes_c);
    if (vk_get_vram_allocation(device_a, &buffer_a, &offset_a, &size_a) != 0 ||
        vk_get_vram_allocation(device_b, &buffer_b, &offset_b, &size_b) != 0 ||
        vk_get_vram_allocation(device_c, &buffer_c, &offset_c, &size_c) != 0) {
        goto cleanup;
    }
    if (buffer_a != buffer_b || buffer_a != buffer_c) {
        report_error("matmul matrices do not share the expected Vulkan pool buffer");
        goto cleanup;
    }

    memset(&descriptor_pool_size, 0, sizeof(descriptor_pool_size));
    descriptor_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_pool_size.descriptorCount = 3;
    memset(&descriptor_pool_info, 0, sizeof(descriptor_pool_info));
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &descriptor_pool_size;
    result = vkCreateDescriptorPool(g_vulkan.device, &descriptor_pool_info,
                                    NULL, &descriptor_pool);
    if (check_vk_result(result, "vkCreateDescriptorPool(matmul)") != 0) {
        goto cleanup;
    }
    memset(&descriptor_allocate_info, 0, sizeof(descriptor_allocate_info));
    descriptor_allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor_allocate_info.descriptorPool = descriptor_pool;
    descriptor_allocate_info.descriptorSetCount = 1;
    descriptor_allocate_info.pSetLayouts = &g_matmul_descriptor_layout;
    result = vkAllocateDescriptorSets(g_vulkan.device, &descriptor_allocate_info,
                                      &descriptor_set);
    if (check_vk_result(result, "vkAllocateDescriptorSets(matmul)") != 0) {
        goto cleanup;
    }

    buffer_infos[0].buffer = buffer_a;
    buffer_infos[0].offset = offset_a;
    buffer_infos[0].range = size_a;
    buffer_infos[1].buffer = buffer_b;
    buffer_infos[1].offset = offset_b;
    buffer_infos[1].range = size_b;
    buffer_infos[2].buffer = buffer_c;
    buffer_infos[2].offset = offset_c;
    buffer_infos[2].range = size_c;
    memset(descriptor_writes, 0, sizeof(descriptor_writes));
    for (uint32_t i = 0; i < 3; ++i) {
        descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[i].dstSet = descriptor_set;
        descriptor_writes[i].dstBinding = i;
        descriptor_writes[i].descriptorCount = 1;
        descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[i].pBufferInfo = &buffer_infos[i];
    }
    vkUpdateDescriptorSets(g_vulkan.device, 3, descriptor_writes, 0, NULL);

    memset(&command_allocate_info, 0, sizeof(command_allocate_info));
    command_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_allocate_info.commandPool = g_vulkan.command_pool;
    command_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_allocate_info.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(g_vulkan.device, &command_allocate_info,
                                      &command_buffer);
    if (check_vk_result(result, "vkAllocateCommandBuffers(matmul)") != 0) {
        goto cleanup;
    }
    command_allocated = 1;
    memset(&command_begin_info, 0, sizeof(command_begin_info));
    command_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = vkBeginCommandBuffer(command_buffer, &command_begin_info);
    if (check_vk_result(result, "vkBeginCommandBuffer(matmul)") != 0) {
        goto cleanup;
    }
    dimensions[0] = (uint32_t)m;
    dimensions[1] = (uint32_t)k;
    dimensions[2] = (uint32_t)n;
    dimensions[3] = (uint32_t)transpose_b |
                    (weight_format == 2 ? UINT32_C(2) : UINT32_C(0));
    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      weight_format != 0 ? g_matmul_bf16_pipeline
                                         : g_matmul_pipeline);
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            g_matmul_pipeline_layout, 0, 1, &descriptor_set,
                            0, NULL);
    vkCmdPushConstants(command_buffer, g_matmul_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(dimensions),
                       dimensions);
    vkCmdDispatch(command_buffer, groups_x, groups_y, 1);
    result = vkEndCommandBuffer(command_buffer);
    if (check_vk_result(result, "vkEndCommandBuffer(matmul)") != 0) {
        goto cleanup;
    }

    memset(&fence_info, 0, sizeof(fence_info));
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    result = vkCreateFence(g_vulkan.device, &fence_info, NULL, &fence);
    if (check_vk_result(result, "vkCreateFence(matmul)") != 0) {
        goto cleanup;
    }
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    result = vkQueueSubmit(g_vulkan.compute_queue, 1, &submit_info, fence);
    if (check_vk_result(result, "vkQueueSubmit(matmul)") != 0) {
        goto cleanup;
    }
    submitted = 1;
    result = vkWaitForFences(g_vulkan.device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (check_vk_result(result, "vkWaitForFences(matmul)") != 0) {
        goto cleanup;
    }
    wait_succeeded = 1;
    memcpy(c, device_c, bytes_c);
    status = 0;

cleanup:
    if (fence != VK_NULL_HANDLE) {
        vkDestroyFence(g_vulkan.device, fence, NULL);
    }
    if (command_allocated) {
        if (submitted && !wait_succeeded) {
            VkResult idle_result = vkDeviceWaitIdle(g_vulkan.device);
            (void)check_vk_result(idle_result, "vkDeviceWaitIdle(matmul cleanup)");
        }
        vkFreeCommandBuffers(g_vulkan.device, g_vulkan.command_pool, 1,
                             &command_buffer);
    }
    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(g_vulkan.device, descriptor_pool, NULL);
    }
    if (device_c != NULL) vk_free_vram(device_c);
    if (device_b != NULL) vk_free_vram(device_b);
    if (device_a != NULL) vk_free_vram(device_a);
    matmul_unlock();
    return status;
}

int vk_matmul_checked(const float *a, const float *b, float *c,
                      int m, int k, int n) {
    return vk_matmul_impl(a, b, c, m, k, n, 0, 0);
}

int vk_matmul_transposed_checked(const float *a, const float *b, float *c,
                                 int m, int k, int n) {
    return vk_matmul_impl(a, b, c, m, k, n, 0, 1);
}

int vk_matmul_bf16_checked(const float *a, const uint16_t *b, float *c,
                           int m, int k, int n) {
    return vk_matmul_impl(a, b, c, m, k, n, 1, 0);
}

int vk_matmul_bf16_transposed_checked(const float *a, const uint16_t *b,
                                      float *c, int m, int k, int n) {
    return vk_matmul_impl(a, b, c, m, k, n, 1, 1);
}

int vk_matmul_fp16_transposed_checked(const float *a, const uint16_t *b,
                                      float *c, int m, int k, int n) {
    return vk_matmul_impl(a, b, c, m, k, n, 2, 1);
}

void vk_matmul(float *a, float *b, float *c, int m, int k, int n) {
    (void)vk_matmul_checked(a, b, c, m, k, n);
}

VkDevice vk_get_device(void) {
    return g_vulkan.initialized ? g_vulkan.device : VK_NULL_HANDLE;
}

VkPhysicalDevice vk_get_physical_device(void) {
    return g_vulkan.initialized ? g_vulkan.physical_device : VK_NULL_HANDLE;
}

VkQueue vk_get_compute_queue(void) {
    return g_vulkan.initialized ? g_vulkan.compute_queue : VK_NULL_HANDLE;
}

uint32_t vk_get_compute_queue_family_index(void) {
    return g_vulkan.initialized ? g_vulkan.queue_family_index : UINT32_MAX;
}

VkCommandPool vk_get_compute_command_pool(void) {
    return g_vulkan.initialized ? g_vulkan.command_pool : VK_NULL_HANDLE;
}
