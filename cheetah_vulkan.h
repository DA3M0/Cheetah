#ifndef CHEETAH_VULKAN_H
#define CHEETAH_VULKAN_H

#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 0 on success and -1 on failure. The pool defaults to 800 MiB;
 * CHEETAH_VRAM_POOL_SIZE may override it with a positive byte count.
 * Initialization and shutdown must not run concurrently with pool operations. */
int init_vulkan_device(void);
void shutdown_vulkan_device(void);

/* Allocations are uninitialized, host-mapped storage-buffer ranges. A null
 * return indicates failure. Pool offsets are aligned to 4 KiB and the device's
 * storage-buffer offset limit. Free only after submitted GPU work has finished. */
void *vk_alloc_vram(size_t size);
void vk_free_vram(void *pointer);

/* Returns 0 on success. The buffer is the shared pool buffer; offset and size
 * describe this allocation and can be used in VkDescriptorBufferInfo. */
int vk_get_vram_allocation(void *pointer, VkBuffer *buffer,
                           VkDeviceSize *offset, VkDeviceSize *size);

/* Shader modules and pipelines are owned by the caller and must be destroyed
 * before shutdown_vulkan_device(). A null handle indicates failure. */
VkShaderModule vk_load_spirv_shader(const char *path);
VkPipeline vk_create_compute_pipeline(VkShaderModule shader,
                                      VkPipelineLayout layout,
                                      const char *entry_point);

/* Computes row-major C[M,N] = A[M,K] * B[K,N]. Requires successful device
 * initialization and matmul.spv in the current directory unless
 * CHEETAH_MATMUL_SPV names another compiled SPIR-V module. Compile the source
 * with: glslc -fshader-stage=compute matmul.comp -o matmul.spv
 * Errors are printed to stderr; c is unchanged on failure. */
void vk_matmul(float *a, float *b, float *c, int m, int k, int n);
/* Status-returning counterpart for language bindings. */
int vk_matmul_checked(const float *a, const float *b, float *c,
                      int m, int k, int n);

/* Handles returned here are borrowed and remain valid until shutdown. */
VkDevice vk_get_device(void);
VkPhysicalDevice vk_get_physical_device(void);
VkQueue vk_get_compute_queue(void);
uint32_t vk_get_compute_queue_family_index(void);
VkCommandPool vk_get_compute_command_pool(void);

#ifdef __cplusplus
}
#endif

#endif
