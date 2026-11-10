#define _POSIX_C_SOURCE 200809L

#include "cheetah_color.h"
#include "cheetah_init.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ALIGNMENT ((size_t)4096)
#define COPY_CHUNK ((size_t)1024 * 1024)
#define MAX_HEADER_SIZE ((size_t)128 * 1024 * 1024)

typedef struct JsonParser {
    const char *cursor;
    const char *end;
} JsonParser;

typedef struct StringBuilder {
    char *data;
    size_t length;
    size_t capacity;
} StringBuilder;

typedef struct TensorRecord {
    char *name;
    char *dtype;
    uint64_t *shape;
    size_t dimensions;
    size_t source_index;
    uint64_t source_file_offset;
    uint64_t payload_size;
    uint64_t shard_offset;
    uint64_t shard_data_offset;
    char *shard_name;
} TensorRecord;

typedef struct SourceRecord {
    char *path;
    unsigned char *mapping;
    size_t mapping_size;
} SourceRecord;

typedef struct ShardRecord {
    char *name;
    char *path;
    uint64_t size;
} ShardRecord;

typedef struct ModelInfo {
    TensorRecord *tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    char **tensor_name_slots;
    size_t tensor_name_slot_count;
    SourceRecord *sources;
    size_t source_count;
    size_t source_capacity;
    ShardRecord *shards;
    size_t shard_count;
    size_t shard_capacity;
    uint64_t num_layers;
    uint64_t hidden_size;
    uint64_t vocab_size;
    uint64_t intermediate_size;
    uint64_t num_attention_heads;
    uint64_t num_key_value_heads;
    uint64_t max_layer_index;
    uint64_t max_context;
    int found_layer;
    int found_embedding;
    int found_intermediate;
    int found_attention;
    int has_q_norm;
    uint64_t q_norm_size;
    int found_query;
    uint64_t query_output_size;
    int found_key;
    uint64_t key_output_size;
    int model_type_from_config;
    int has_im_start;
    int has_im_end;
    int has_header_tokens;
    int has_eot_id;
    int has_start_of_turn;
    int has_end_of_turn;
    int has_phi_end;
    int has_fim_suffix;
    int has_endoftext;
    int has_end_of_text;
    int has_eos;
    char model_type[32];
    char *chat_template;
    const char *stop_markers[8];
    size_t stop_marker_count;
    cheetah_progress_fn progress_fn;
    void *progress_user_data;
} ModelInfo;

static void set_error(const char *message) {
    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED, "cheetah-init: %s\n", message);
}

static void set_path_error(const char *operation, const char *path) {
    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                    "cheetah-init: %s '%s': %s\n", operation, path,
                    strerror(errno));
}

static int builder_reserve(StringBuilder *builder, size_t additional) {
    size_t needed;
    size_t capacity;
    char *data;

    if (additional > SIZE_MAX - builder->length - 1) {
        return -1;
    }
    needed = builder->length + additional + 1;
    if (needed <= builder->capacity) {
        return 0;
    }
    capacity = builder->capacity == 0 ? 128 : builder->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    data = (char *)realloc(builder->data, capacity);
    if (data == NULL) {
        return -1;
    }
    builder->data = data;
    builder->capacity = capacity;
    return 0;
}

static int builder_append_n(StringBuilder *builder, const char *text,
                            size_t length) {
    if (builder_reserve(builder, length) != 0) {
        return -1;
    }
    memcpy(builder->data + builder->length, text, length);
    builder->length += length;
    builder->data[builder->length] = '\0';
    return 0;
}

static int builder_append(StringBuilder *builder, const char *text) {
    return builder_append_n(builder, text, strlen(text));
}

static void skip_space(JsonParser *parser) {
    while (parser->cursor < parser->end &&
           (*parser->cursor == ' ' || *parser->cursor == '\t' ||
            *parser->cursor == '\n' || *parser->cursor == '\r')) {
        ++parser->cursor;
    }
}

static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int append_utf8(StringBuilder *builder, uint32_t codepoint) {
    char bytes[4];
    size_t length;

    if (codepoint <= 0x7f) {
        bytes[0] = (char)codepoint;
        length = 1;
    } else if (codepoint <= 0x7ff) {
        bytes[0] = (char)(0xc0 | (codepoint >> 6));
        bytes[1] = (char)(0x80 | (codepoint & 0x3f));
        length = 2;
    } else if (codepoint <= 0xffff) {
        bytes[0] = (char)(0xe0 | (codepoint >> 12));
        bytes[1] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[2] = (char)(0x80 | (codepoint & 0x3f));
        length = 3;
    } else if (codepoint <= 0x10ffff) {
        bytes[0] = (char)(0xf0 | (codepoint >> 18));
        bytes[1] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        bytes[2] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        bytes[3] = (char)(0x80 | (codepoint & 0x3f));
        length = 4;
    } else {
        return -1;
    }
    return builder_append_n(builder, bytes, length);
}

static int parse_hex4(JsonParser *parser, uint32_t *value) {
    unsigned i;
    uint32_t result = 0;

    if ((size_t)(parser->end - parser->cursor) < 4) {
        return -1;
    }
    for (i = 0; i < 4; ++i) {
        int digit = hex_value(parser->cursor[i]);
        if (digit < 0) {
            return -1;
        }
        result = (result << 4) | (uint32_t)digit;
    }
    parser->cursor += 4;
    *value = result;
    return 0;
}

static char *parse_json_string(JsonParser *parser) {
    StringBuilder builder = {0};

    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '"') {
        return NULL;
    }
    while (parser->cursor < parser->end) {
        unsigned char c = (unsigned char)*parser->cursor++;
        if (c == '"') {
            if (builder.data == NULL) {
                builder.data = (char *)calloc(1, 1);
            }
            return builder.data;
        }
        if (c < 0x20) {
            break;
        }
        if (c != '\\') {
            if (builder_append_n(&builder, (const char *)&c, 1) != 0) {
                break;
            }
            continue;
        }
        if (parser->cursor >= parser->end) {
            break;
        }
        c = (unsigned char)*parser->cursor++;
        switch (c) {
        case '"': case '\\': case '/':
            if (builder_append_n(&builder, (const char *)&c, 1) != 0) goto fail;
            break;
        case 'b': c = '\b'; goto append_escape;
        case 'f': c = '\f'; goto append_escape;
        case 'n': c = '\n'; goto append_escape;
        case 'r': c = '\r'; goto append_escape;
        case 't': c = '\t';
append_escape:
            if (builder_append_n(&builder, (const char *)&c, 1) != 0) goto fail;
            break;
        case 'u': {
            uint32_t first;
            if (parse_hex4(parser, &first) != 0) goto fail;
            if (first >= 0xd800 && first <= 0xdbff) {
                uint32_t second;
                if ((size_t)(parser->end - parser->cursor) < 6 ||
                    parser->cursor[0] != '\\' || parser->cursor[1] != 'u') {
                    goto fail;
                }
                parser->cursor += 2;
                if (parse_hex4(parser, &second) != 0 ||
                    second < 0xdc00 || second > 0xdfff) {
                    goto fail;
                }
                first = 0x10000 + ((first - 0xd800) << 10) +
                        (second - 0xdc00);
            }
            if (first >= 0xdc00 && first <= 0xdfff) goto fail;
            if (append_utf8(&builder, first) != 0) goto fail;
            break;
        }
        default:
            goto fail;
        }
    }
fail:
    free(builder.data);
    return NULL;
}

static int parse_uint64(JsonParser *parser, uint64_t *value) {
    uint64_t number = 0;
    const char *start;

    skip_space(parser);
    start = parser->cursor;
    if (start >= parser->end || *start < '0' || *start > '9') {
        return -1;
    }
    while (parser->cursor < parser->end &&
           *parser->cursor >= '0' && *parser->cursor <= '9') {
        unsigned digit = (unsigned)(*parser->cursor++ - '0');
        if (number > (UINT64_MAX - digit) / 10) {
            return -1;
        }
        number = number * 10 + digit;
    }
    if (parser->cursor < parser->end &&
        (*parser->cursor == '.' || *parser->cursor == 'e' ||
         *parser->cursor == 'E' || *parser->cursor == '-')) {
        return -1;
    }
    *value = number;
    return 0;
}

static int skip_json_value(JsonParser *parser);

static int skip_json_array(JsonParser *parser) {
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '[') return -1;
    skip_space(parser);
    if (parser->cursor < parser->end && *parser->cursor == ']') {
        ++parser->cursor;
        return 0;
    }
    for (;;) {
        if (skip_json_value(parser) != 0) return -1;
        skip_space(parser);
        if (parser->cursor >= parser->end) return -1;
        if (*parser->cursor == ']') {
            ++parser->cursor;
            return 0;
        }
        if (*parser->cursor++ != ',') return -1;
    }
}

static int skip_json_object(JsonParser *parser) {
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '{') return -1;
    skip_space(parser);
    if (parser->cursor < parser->end && *parser->cursor == '}') {
        ++parser->cursor;
        return 0;
    }
    for (;;) {
        char *key = parse_json_string(parser);
        if (key == NULL) return -1;
        free(key);
        skip_space(parser);
        if (parser->cursor >= parser->end || *parser->cursor++ != ':') return -1;
        if (skip_json_value(parser) != 0) return -1;
        skip_space(parser);
        if (parser->cursor >= parser->end) return -1;
        if (*parser->cursor == '}') {
            ++parser->cursor;
            return 0;
        }
        if (*parser->cursor++ != ',') return -1;
    }
}

static int skip_json_value(JsonParser *parser) {
    skip_space(parser);
    if (parser->cursor >= parser->end) return -1;
    switch (*parser->cursor) {
    case '"': {
        char *string = parse_json_string(parser);
        if (string == NULL) return -1;
        free(string);
        return 0;
    }
    case '{': return skip_json_object(parser);
    case '[': return skip_json_array(parser);
    case 't':
        if ((size_t)(parser->end - parser->cursor) >= 4 &&
            memcmp(parser->cursor, "true", 4) == 0) {
            parser->cursor += 4;
            return 0;
        }
        return -1;
    case 'f':
        if ((size_t)(parser->end - parser->cursor) >= 5 &&
            memcmp(parser->cursor, "false", 5) == 0) {
            parser->cursor += 5;
            return 0;
        }
        return -1;
    case 'n':
        if ((size_t)(parser->end - parser->cursor) >= 4 &&
            memcmp(parser->cursor, "null", 4) == 0) {
            parser->cursor += 4;
            return 0;
        }
        return -1;
    default: {
        uint64_t ignored;
        return parse_uint64(parser, &ignored);
    }
    }
}

static int parse_uint_array(JsonParser *parser, uint64_t **values_out,
                            size_t *count_out) {
    uint64_t *values = NULL;
    size_t count = 0;
    size_t capacity = 0;

    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '[') return -1;
    skip_space(parser);
    while (parser->cursor < parser->end && *parser->cursor != ']') {
        uint64_t value;
        uint64_t *grown;
        if (parse_uint64(parser, &value) != 0) {
            free(values);
            return -1;
        }
        if (count == capacity) {
            size_t next = capacity == 0 ? 4 : capacity * 2;
            if (next < capacity || next > SIZE_MAX / sizeof(*values)) {
                free(values);
                return -1;
            }
            grown = (uint64_t *)realloc(values, next * sizeof(*values));
            if (grown == NULL) {
                free(values);
                return -1;
            }
            values = grown;
            capacity = next;
        }
        values[count++] = value;
        skip_space(parser);
        if (parser->cursor < parser->end && *parser->cursor == ']') break;
        if (parser->cursor >= parser->end || *parser->cursor++ != ',') {
            free(values);
            return -1;
        }
    }
    if (parser->cursor >= parser->end || *parser->cursor++ != ']') {
        free(values);
        return -1;
    }
    *values_out = values;
    *count_out = count;
    return 0;
}

static int parse_tensor_object(JsonParser *parser, TensorRecord *tensor) {
    int has_dtype = 0;
    int has_shape = 0;
    int has_offsets = 0;

    memset(tensor, 0, sizeof(*tensor));
    skip_space(parser);
    if (parser->cursor >= parser->end || *parser->cursor++ != '{') return -1;
    skip_space(parser);
    while (parser->cursor < parser->end && *parser->cursor != '}') {
        char *key = parse_json_string(parser);
        if (key == NULL) goto fail;
        skip_space(parser);
        if (parser->cursor >= parser->end || *parser->cursor++ != ':') {
            free(key);
            goto fail;
        }
        if (strcmp(key, "dtype") == 0) {
            tensor->dtype = parse_json_string(parser);
            if (tensor->dtype == NULL) {
                free(key);
                goto fail;
            }
            has_dtype = 1;
        } else if (strcmp(key, "shape") == 0) {
            if (parse_uint_array(parser, &tensor->shape,
                                 &tensor->dimensions) != 0) {
                free(key);
                goto fail;
            }
            has_shape = 1;
        } else if (strcmp(key, "data_offsets") == 0) {
            uint64_t *offsets = NULL;
            size_t count = 0;
            if (parse_uint_array(parser, &offsets, &count) != 0 ||
                count != 2 || offsets[1] < offsets[0]) {
                free(offsets);
                free(key);
                goto fail;
            }
            tensor->source_file_offset = offsets[0];
            tensor->payload_size = offsets[1] - offsets[0];
            free(offsets);
            has_offsets = 1;
        } else if (skip_json_value(parser) != 0) {
            free(key);
            goto fail;
        }
        free(key);
        skip_space(parser);
        if (parser->cursor < parser->end && *parser->cursor == '}') break;
        if (parser->cursor >= parser->end || *parser->cursor++ != ',') goto fail;
        skip_space(parser);
    }
    if (parser->cursor >= parser->end || *parser->cursor++ != '}' ||
        !has_dtype || !has_shape || !has_offsets) {
        goto fail;
    }
    return 0;
fail:
    free(tensor->dtype);
    free(tensor->shape);
    tensor->dtype = NULL;
    tensor->shape = NULL;
    return -1;
}

static int grow_array(void **array, size_t *capacity, size_t count,
                      size_t element_size) {
    size_t next = *capacity == 0 ? 16 : *capacity * 2;
    void *grown;

    if (count < *capacity) return 0;
    if (next < *capacity || next > SIZE_MAX / element_size) return -1;
    grown = realloc(*array, next * element_size);
    if (grown == NULL) return -1;
    *array = grown;
    *capacity = next;
    return 0;
}

static int parse_safetensors_header(const unsigned char *mapping,
                                   size_t mapping_size, uint64_t *data_start,
                                   TensorRecord **tensors_out,
                                   size_t *count_out) {
    uint64_t header_size = 0;
    JsonParser parser;
    TensorRecord *tensors = NULL;
    size_t count = 0;
    size_t capacity = 0;
    size_t i;

    if (mapping_size < 8) {
        set_error("Safetensors file is shorter than its 8-byte header length");
        return -1;
    }
    for (i = 0; i < 8; ++i) header_size |= (uint64_t)mapping[i] << (8 * i);
    if (header_size > MAX_HEADER_SIZE || header_size > SIZE_MAX - 8 ||
        (size_t)header_size > mapping_size - 8) {
        set_error("Safetensors header length is invalid or exceeds 128 MiB");
        return -1;
    }
    *data_start = 8 + header_size;
    parser.cursor = (const char *)mapping + 8;
    parser.end = parser.cursor + (size_t)header_size;
    skip_space(&parser);
    if (parser.cursor >= parser.end || *parser.cursor++ != '{') goto invalid;
    skip_space(&parser);
    while (parser.cursor < parser.end && *parser.cursor != '}') {
        char *name = parse_json_string(&parser);
        TensorRecord tensor;
        if (name == NULL) goto invalid;
        skip_space(&parser);
        if (parser.cursor >= parser.end || *parser.cursor++ != ':') {
            free(name);
            goto invalid;
        }
        skip_space(&parser);
        if (strcmp(name, "__metadata__") == 0) {
            free(name);
            if (skip_json_value(&parser) != 0) goto invalid;
        } else {
            if (parse_tensor_object(&parser, &tensor) != 0) {
                free(name);
                goto invalid;
            }
            tensor.name = name;
            if (grow_array((void **)&tensors, &capacity, count,
                           sizeof(*tensors)) != 0) {
                free(tensor.name);
                free(tensor.dtype);
                free(tensor.shape);
                goto no_memory;
            }
            tensors[count++] = tensor;
        }
        skip_space(&parser);
        if (parser.cursor < parser.end && *parser.cursor == '}') break;
        if (parser.cursor >= parser.end || *parser.cursor++ != ',') goto invalid;
        skip_space(&parser);
    }
    if (parser.cursor >= parser.end || *parser.cursor++ != '}') goto invalid;
    skip_space(&parser);
    if (parser.cursor != parser.end || count == 0) goto invalid;
    *tensors_out = tensors;
    *count_out = count;
    return 0;

no_memory:
    set_error("out of memory while parsing Safetensors header");
    goto cleanup;
invalid:
    set_error("invalid Safetensors JSON header or tensor metadata");
cleanup:
    for (i = 0; i < count; ++i) {
        free(tensors[i].name);
        free(tensors[i].dtype);
        free(tensors[i].shape);
    }
    free(tensors);
    return -1;
}

static int path_join(const char *left, const char *right, char **out) {
    size_t left_len = strlen(left);
    size_t right_len = strlen(right);
    int separator = left_len > 0 && left[left_len - 1] != '/';
    char *joined;

    if (left_len > SIZE_MAX - right_len - (size_t)separator - 1) return -1;
    joined = (char *)malloc(left_len + right_len + (size_t)separator + 1);
    if (joined == NULL) return -1;
    memcpy(joined, left, left_len);
    if (separator) joined[left_len++] = '/';
    memcpy(joined + left_len, right, right_len);
    joined[left_len + right_len] = '\0';
    *out = joined;
    return 0;
}

static int make_directories(const char *path) {
    char *copy = strdup(path);
    char *cursor;
    struct stat status;

    if (copy == NULL) return -1;
    for (cursor = copy + (copy[0] == '/' ? 1 : 0); ; ++cursor) {
        if (*cursor != '/' && *cursor != '\0') continue;
        {
            char saved = *cursor;
            *cursor = '\0';
            if (copy[0] != '\0' && mkdir(copy, 0755) != 0 &&
                errno != EEXIST) {
                set_path_error("cannot create directory", copy);
                free(copy);
                return -1;
            }
            if (stat(copy, &status) != 0 || !S_ISDIR(status.st_mode)) {
                CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                                "cheetah-init: '%s' is not a directory\n", copy);
                free(copy);
                return -1;
            }
            *cursor = saved;
            if (saved == '\0') break;
        }
    }
    free(copy);
    return 0;
}

static int string_compare(const void *left, const void *right) {
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return strcmp(*a, *b);
}

static int ends_with_safetensors(const char *name) {
    size_t length = strlen(name);
    return length >= 12 && strcmp(name + length - 12, ".safetensors") == 0;
}

static int copy_file(const char *source, const char *destination);

static int ensure_copy(const char *source, const char *destination) {
    struct stat status;
    if (stat(destination, &status) == 0) return 0;
    if (errno != ENOENT) {
        set_path_error("cannot stat destination file", destination);
        return -1;
    }
    if (source == NULL || stat(source, &status) != 0 ||
        !S_ISREG(status.st_mode)) {
        if (source != NULL && errno != ENOENT) {
            set_path_error("cannot stat source file", source);
            return -1;
        }
        return 0;
    }
    return copy_file(source, destination);
}

static int collect_source_directory(const char *directory_path,
                                    char ***files, size_t *count,
                                    size_t *capacity) {
    DIR *directory = opendir(directory_path);
    struct dirent *entry;
    if (directory == NULL) {
        set_path_error("cannot open source directory", directory_path);
        return -1;
    }
    while ((entry = readdir(directory)) != NULL) {
        char *path;
        struct stat status;
        int is_symlink;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0 ||
            strcmp(entry->d_name, ".cheetah") == 0) {
            continue;
        }
        if (path_join(directory_path, entry->d_name, &path) != 0) {
            closedir(directory);
            set_error("out of memory while joining a source path");
            return -1;
        }
        if (lstat(path, &status) != 0) {
            free(path);
            continue;
        }
        is_symlink = S_ISLNK(status.st_mode);
        if (is_symlink &&
            (stat(path, &status) != 0 || S_ISDIR(status.st_mode))) {
            free(path);
            continue;
        }
        if (S_ISDIR(status.st_mode)) {
            if (collect_source_directory(path, files, count, capacity) != 0) {
                free(path);
                closedir(directory);
                return -1;
            }
        } else if (S_ISREG(status.st_mode) &&
                   ends_with_safetensors(entry->d_name)) {
            if (grow_array((void **)files, capacity, *count,
                           sizeof(**files)) != 0) {
                free(path);
                closedir(directory);
                set_error("out of memory while collecting source files");
                return -1;
            }
            (*files)[(*count)++] = path;
            path = NULL;
        }
        free(path);
    }
    closedir(directory);
    return 0;
}

static int collect_source_files(const char *source, char ***files_out,
                                size_t *count_out) {
    struct stat status;
    char **files = NULL;
    size_t count = 0;
    size_t capacity = 0;

    if (stat(source, &status) != 0) {
        set_path_error("cannot stat source", source);
        return -1;
    }
    if (S_ISREG(status.st_mode)) {
        if (!ends_with_safetensors(source)) {
            set_error("source file must have a .safetensors extension");
            return -1;
        }
        files = (char **)malloc(sizeof(*files));
        if (files == NULL || (files[0] = strdup(source)) == NULL) {
            free(files);
            set_error("out of memory while collecting source files");
            return -1;
        }
        *files_out = files;
        *count_out = 1;
        return 0;
    }
    if (!S_ISDIR(status.st_mode)) {
        set_error("source must be a directory or a regular Safetensors file");
        return -1;
    }
    if (collect_source_directory(source, &files, &count, &capacity) != 0)
        goto fail;
    if (count == 0) {
        set_error("source directory contains no top-level .safetensors files");
        goto fail;
    }
    qsort(files, count, sizeof(*files), string_compare);
    *files_out = files;
    *count_out = count;
    return 0;
fail:
    while (count > 0) free(files[--count]);
    free(files);
    return -1;
}

static int parse_layer_number(const char *name, uint64_t *layer_out) {
    const char *cursor = strstr(name, "layers.");
    char *end;
    unsigned long long parsed;

    if (cursor == NULL) cursor = strstr(name, "h.");
    if (cursor == NULL) return 0;
    cursor = strchr(cursor, '.') + 1;
    if (*cursor < '0' || *cursor > '9') return 0;
    errno = 0;
    parsed = strtoull(cursor, &end, 10);
    if (errno != 0 || end == cursor) return 0;
    *layer_out = (uint64_t)parsed;
    return 1;
}

static char *make_shard_name(const char *tensor_name) {
    uint64_t layer = 0;
    int has_layer = parse_layer_number(tensor_name, &layer);
    char name[128];

    if (has_layer) {
        (void)snprintf(name, sizeof(name), "layer_%" PRIu64 ".safetensors",
                       layer);
    } else {
        (void)snprintf(name, sizeof(name), "global.safetensors");
    }
    return strdup(name);
}

static uint64_t hash_string(const char *text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    while (*text != '\0') {
        hash ^= (unsigned char)*text++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int insert_tensor_name(ModelInfo *model, const char *name) {
    size_t capacity = model->tensor_name_slot_count;
    char **slots = model->tensor_name_slots;
    size_t i;

    if (capacity == 0 || model->tensor_count + 1 > capacity / 2) {
        size_t next = capacity == 0 ? 128 : capacity * 2;
        char **grown;
        if (next < capacity || next > SIZE_MAX / sizeof(*grown)) return -1;
        grown = (char **)calloc(next, sizeof(*grown));
        if (grown == NULL) return -1;
        for (i = 0; i < capacity; ++i) {
            if (slots[i] != NULL) {
                size_t index = (size_t)hash_string(slots[i]) & (next - 1);
                while (grown[index] != NULL) index = (index + 1) & (next - 1);
                grown[index] = slots[i];
            }
        }
        free(slots);
        model->tensor_name_slots = grown;
        model->tensor_name_slot_count = next;
        slots = grown;
        capacity = next;
    }
    i = (size_t)hash_string(name) & (capacity - 1);
    while (slots[i] != NULL) {
        if (strcmp(slots[i], name) == 0) return 1;
        i = (i + 1) & (capacity - 1);
    }
    slots[i] = (char *)name;
    return 0;
}

static int add_tensor_record(ModelInfo *model, TensorRecord *tensor) {
    int name_result;

    if (grow_array((void **)&model->tensors, &model->tensor_capacity,
                   model->tensor_count, sizeof(*model->tensors)) != 0) {
        set_error("out of memory while recording tensor metadata");
        return -1;
    }
    name_result = insert_tensor_name(model, tensor->name);
    if (name_result == 1) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                        "cheetah-init: duplicate tensor name '%s'\n",
                        tensor->name);
        return -1;
    }
    if (name_result != 0) {
        set_error("out of memory while recording tensor metadata");
        return -1;
    }
    model->tensors[model->tensor_count++] = *tensor;
    memset(tensor, 0, sizeof(*tensor));
    return 0;
}

static ShardRecord *get_shard(ModelInfo *model, const char *name,
                              const char *weights_dir) {
    size_t i;
    ShardRecord *shard;
    char *path;

    for (i = 0; i < model->shard_count; ++i) {
        if (strcmp(model->shards[i].name, name) == 0) {
            return &model->shards[i];
        }
    }
    if (grow_array((void **)&model->shards, &model->shard_capacity,
                   model->shard_count, sizeof(*model->shards)) != 0) {
        set_error("out of memory while recording shard metadata");
        return NULL;
    }
    if (path_join(weights_dir, name, &path) != 0) {
        set_error("out of memory while joining a shard path");
        return NULL;
    }
    shard = &model->shards[model->shard_count];
    memset(shard, 0, sizeof(*shard));
    shard->name = strdup(name);
    shard->path = path;
    if (shard->name == NULL) {
        free(shard->name);
        free(shard->path);
        memset(shard, 0, sizeof(*shard));
        return NULL;
    }
    ++model->shard_count;
    return shard;
}

static void update_inference(ModelInfo *model, const TensorRecord *tensor) {
    uint64_t layer;
    const char *name = tensor->name;
    const uint64_t *shape = tensor->shape;

    if (parse_layer_number(name, &layer)) {
        model->found_layer = 1;
        if (layer > model->max_layer_index) model->max_layer_index = layer;
    }
    if (tensor->dimensions == 2 &&
        (strcmp(name, "model.embed_tokens.weight") == 0 ||
         strcmp(name, "transformer.wte.weight") == 0 ||
         strcmp(name, "tok_embeddings.weight") == 0 ||
         strcmp(name, "model.word_embeddings.weight") == 0)) {
        model->vocab_size = shape[0];
        model->hidden_size = shape[1];
        model->found_embedding = 1;
    }
    if (tensor->dimensions == 2 &&
        (strstr(name, ".mlp.gate_proj.weight") != NULL ||
         strstr(name, ".mlp.up_proj.weight") != NULL ||
         strstr(name, ".mlp.fc1.weight") != NULL)) {
        model->intermediate_size = shape[0];
        model->found_intermediate = 1;
    }
    if (tensor->dimensions == 1 && strstr(name, ".self_attn.q_norm.weight")) {
        model->q_norm_size = shape[0];
        model->has_q_norm = 1;
    }
    if (tensor->dimensions == 2 &&
        strstr(name, ".self_attn.q_proj.weight") != NULL) {
        model->query_output_size = shape[0];
        model->found_query = 1;
        model->found_attention = 1;
    }
    if (tensor->dimensions == 2 &&
        strstr(name, ".self_attn.k_proj.weight") != NULL) {
        model->key_output_size = shape[0];
        model->found_key = 1;
    }
    if (model->model_type_from_config) return;
    if (strstr(name, "transformer.h.") != NULL) {
        (void)snprintf(model->model_type, sizeof(model->model_type), "phi");
    } else if (strstr(name, "model.layers.") != NULL &&
               strcmp(model->model_type, "phi") != 0 &&
               model->model_type[0] == '\0') {
        (void)snprintf(model->model_type, sizeof(model->model_type), "llama");
    }
}

static void finalize_inference(ModelInfo *model) {
    uint64_t head_dim = 0;

    if (model->found_layer) model->num_layers = model->max_layer_index + 1;
    if (model->has_q_norm && model->q_norm_size > 0 &&
        model->found_query && model->query_output_size % model->q_norm_size == 0) {
        model->num_attention_heads =
            model->query_output_size / model->q_norm_size;
        head_dim = model->q_norm_size;
    } else if (model->found_embedding && model->hidden_size > 0) {
        head_dim = model->hidden_size >= 2048 &&
                           model->hidden_size % 128 == 0
                       ? 128
                       : 64;
        model->num_attention_heads = model->hidden_size / head_dim;
        if (model->num_attention_heads == 0) model->num_attention_heads = 1;
        if (model->found_query && model->query_output_size != 0 &&
            model->query_output_size % head_dim == 0) {
            model->num_attention_heads = model->query_output_size / head_dim;
        }
    }
    if (model->found_key && head_dim > 0 &&
        model->key_output_size % head_dim == 0) {
        model->num_key_value_heads = model->key_output_size / head_dim;
    } else {
        model->num_key_value_heads = model->num_attention_heads;
    }
    if (model->num_key_value_heads == 0) {
        model->num_key_value_heads = model->num_attention_heads;
    }
    model->max_context = 4096;
    if (model->model_type[0] == '\0') {
        (void)snprintf(model->model_type, sizeof(model->model_type), "llama");
    }
}

static int add_source_mapping(ModelInfo *model, const char *path,
                              unsigned char *mapping, size_t mapping_size,
                              size_t *source_index_out) {
    SourceRecord *source;

    if (grow_array((void **)&model->sources, &model->source_capacity,
                   model->source_count, sizeof(*model->sources)) != 0) {
        set_error("out of memory while recording mapped source files");
        return -1;
    }
    source = &model->sources[model->source_count];
    source->path = strdup(path);
    if (source->path == NULL) {
        set_error("out of memory while recording mapped source path");
        return -1;
    }
    source->mapping = mapping;
    source->mapping_size = mapping_size;
    *source_index_out = model->source_count++;
    return 0;
}

static int process_safetensors_file(const char *path, ModelInfo *model) {
    int fd = -1;
    struct stat status;
    unsigned char *mapping = MAP_FAILED;
    size_t mapping_size;
    uint64_t data_start;
    TensorRecord *tensors = NULL;
    size_t tensor_count = 0;
    size_t source_index = 0;
    size_t i;
    int result = -1;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_path_error("cannot open Safetensors file", path);
        goto done;
    }
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0 || (uintmax_t)status.st_size > SIZE_MAX) {
        set_path_error("invalid Safetensors file", path);
        goto done;
    }
    mapping_size = (size_t)status.st_size;
    mapping = (unsigned char *)mmap(NULL, mapping_size, PROT_READ, MAP_PRIVATE,
                                    fd, 0);
    if (mapping == MAP_FAILED) {
        set_path_error("cannot mmap Safetensors file", path);
        goto done;
    }
    if (parse_safetensors_header(mapping, mapping_size, &data_start,
                                 &tensors, &tensor_count) != 0) {
        goto done;
    }
    if (add_source_mapping(model, path, mapping, mapping_size,
                           &source_index) != 0) {
        goto done;
    }
    mapping = MAP_FAILED;

    for (i = 0; i < tensor_count; ++i) {
        TensorRecord *tensor = &tensors[i];
        uint64_t file_payload;
        uint64_t end;
        uint64_t layer;
        char *shard_name;
        unsigned d;

        if (tensor->source_file_offset > UINT64_MAX - data_start) {
            set_error("tensor payload offset overflows");
            goto done;
        }
        file_payload = data_start + tensor->source_file_offset;
        if (tensor->payload_size > UINT64_MAX - file_payload) {
            set_error("tensor payload size overflows");
            goto done;
        }
        end = file_payload + tensor->payload_size;
        if (end > mapping_size) {
            CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                            "cheetah-init: tensor '%s' extends beyond '%s'\n",
                            tensor->name, path);
            goto done;
        }
        if (parse_layer_number(tensor->name, &layer) &&
            layer == UINT64_MAX) {
            set_error("layer index is out of range");
            goto done;
        }
        for (d = 0; d < tensor->dimensions; ++d) {
            if (tensor->shape[d] == 0) {
                CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_RED,
                                "cheetah-init: tensor '%s' has a zero dimension\n",
                                tensor->name);
                goto done;
            }
        }
        shard_name = make_shard_name(tensor->name);
        if (shard_name == NULL) {
            set_error("out of memory while naming tensor shard");
            goto done;
        }
        tensor->source_index = source_index;
        tensor->source_file_offset = file_payload;
        tensor->shard_name = shard_name;
        update_inference(model, tensor);
        if (add_tensor_record(model, tensor) != 0) goto done;
    }
    result = 0;
done:
    if (tensors != NULL) {
        for (i = 0; i < tensor_count; ++i) {
            free(tensors[i].name);
            free(tensors[i].dtype);
            free(tensors[i].shape);
            free(tensors[i].shard_name);
        }
        free(tensors);
    }
    if (mapping != MAP_FAILED) munmap(mapping, mapping_size);
    if (fd >= 0) close(fd);
    return result;
}

static int builder_appendf(StringBuilder *builder, const char *format, ...) {
    va_list arguments;
    va_list copy;
    int count;

    va_start(arguments, format);
    va_copy(copy, arguments);
    count = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (count < 0 || builder_reserve(builder, (size_t)count) != 0) {
        va_end(arguments);
        return -1;
    }
    (void)vsnprintf(builder->data + builder->length,
                    builder->capacity - builder->length, format, arguments);
    va_end(arguments);
    builder->length += (size_t)count;
    return 0;
}

static int builder_append_json_string(StringBuilder *builder,
                                      const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    if (builder_append(builder, "\"") != 0) return -1;
    while (*cursor != '\0') {
        switch (*cursor) {
        case '"': if (builder_append(builder, "\\\"") != 0) return -1; break;
        case '\\': if (builder_append(builder, "\\\\") != 0) return -1; break;
        case '\b': if (builder_append(builder, "\\b") != 0) return -1; break;
        case '\f': if (builder_append(builder, "\\f") != 0) return -1; break;
        case '\n': if (builder_append(builder, "\\n") != 0) return -1; break;
        case '\r': if (builder_append(builder, "\\r") != 0) return -1; break;
        case '\t': if (builder_append(builder, "\\t") != 0) return -1; break;
        default:
            if (*cursor < 0x20) {
                if (builder_appendf(builder, "\\u%04x", (unsigned)*cursor) != 0)
                    return -1;
            } else if (builder_append_n(builder, (const char *)cursor, 1) != 0) {
                return -1;
            }
        }
        ++cursor;
    }
    return builder_append(builder, "\"");
}

static int tensor_name_exists(const ModelInfo *model, const char *name) {
    size_t i;
    for (i = 0; i < model->tensor_count; ++i) {
        if (strcmp(model->tensors[i].name, name) == 0) return 1;
    }
    return 0;
}

static int append_safetensors_tensor_header(StringBuilder *header,
                                            const char *name,
                                            const char *dtype,
                                            const uint64_t *shape,
                                            size_t dimensions,
                                            uint64_t start, uint64_t end) {
    size_t i;
    if (builder_append_json_string(header, name) != 0 ||
        builder_append(header, ":{\"dtype\":") != 0 ||
        builder_append_json_string(header, dtype) != 0 ||
        builder_append(header, ",\"shape\":[") != 0) {
        return -1;
    }
    for (i = 0; i < dimensions; ++i) {
        if ((i > 0 && builder_append(header, ",") != 0) ||
            builder_appendf(header, "%" PRIu64, shape[i]) != 0) {
            return -1;
        }
    }
    return builder_appendf(header,
                           "],\"data_offsets\":[%" PRIu64 ",%" PRIu64 "]}",
                           start, end);
}

static int write_all(FILE *file, const void *data, size_t size) {
    const unsigned char *cursor = (const unsigned char *)data;
    while (size > 0) {
        size_t written = fwrite(cursor, 1, size, file);
        if (written == 0) return -1;
        cursor += written;
        size -= written;
    }
    return 0;
}

static int write_zero_bytes(FILE *file, uint64_t count,
                            unsigned char *aligned_buffer) {
    memset(aligned_buffer, 0, COPY_CHUNK);
    while (count > 0) {
        size_t chunk = count > COPY_CHUNK ? COPY_CHUNK : (size_t)count;
        if (write_all(file, aligned_buffer, chunk) != 0) return -1;
        count -= chunk;
    }
    return 0;
}

static int write_space_bytes(FILE *file, uint64_t count,
                             unsigned char *aligned_buffer) {
    memset(aligned_buffer, ' ', COPY_CHUNK);
    while (count > 0) {
        size_t chunk = count > COPY_CHUNK ? COPY_CHUNK : (size_t)count;
        if (write_all(file, aligned_buffer, chunk) != 0) return -1;
        count -= chunk;
    }
    return 0;
}

static int write_mapped_payload(FILE *file, const TensorRecord *tensor,
                                const ModelInfo *model,
                                unsigned char *aligned_buffer) {
    const SourceRecord *source;
    uint64_t copied = 0;

    if (tensor->source_index >= model->source_count) {
        set_error("tensor references an invalid mapped source file");
        return -1;
    }
    source = &model->sources[tensor->source_index];
    if (tensor->source_file_offset > source->mapping_size ||
        tensor->payload_size > source->mapping_size - tensor->source_file_offset) {
        set_error("tensor payload range exceeds its mapped source file");
        return -1;
    }
    while (copied < tensor->payload_size) {
        size_t chunk = tensor->payload_size - copied > COPY_CHUNK
                           ? COPY_CHUNK
                           : (size_t)(tensor->payload_size - copied);
        memcpy(aligned_buffer,
               source->mapping + tensor->source_file_offset + copied, chunk);
        if (write_all(file, aligned_buffer, chunk) != 0) return -1;
        copied += chunk;
    }
    return 0;
}

static int write_safetensors_shard(ModelInfo *model, size_t shard_index,
                                   unsigned char *aligned_buffer) {
    ShardRecord *shard = &model->shards[shard_index];
    StringBuilder header = {0};
    FILE *file = NULL;
    size_t i;
    size_t member_count = 0;
    unsigned padding_index = 0;
    uint64_t data_cursor = 0;
    uint64_t data_start;
    uint64_t header_size;
    unsigned char encoded_length[8];
    int failed = 0;

    if (builder_append(&header, "{") != 0) goto no_memory;
    for (i = 0; i < model->tensor_count; ++i) {
        TensorRecord *tensor = &model->tensors[i];
        uint64_t padding;
        if (strcmp(tensor->shard_name, shard->name) != 0) continue;
        padding = (ALIGNMENT - (data_cursor % ALIGNMENT)) % ALIGNMENT;
        if (padding > 0) {
            char padding_name[64];
            uint64_t padding_shape[1] = {padding};
            do {
                (void)snprintf(padding_name, sizeof(padding_name),
                               "__cheetah_padding_%06u", padding_index++);
            } while (tensor_name_exists(model, padding_name));
            if (member_count++ > 0 && builder_append(&header, ",") != 0)
                goto no_memory;
            if (append_safetensors_tensor_header(
                    &header, padding_name, "U8", padding_shape, 1,
                    data_cursor, data_cursor + padding) != 0) {
                goto no_memory;
            }
            data_cursor += padding;
        }
        if (member_count++ > 0 && builder_append(&header, ",") != 0)
            goto no_memory;
        tensor->shard_data_offset = data_cursor;
        if (append_safetensors_tensor_header(
                &header, tensor->name, tensor->dtype, tensor->shape,
                tensor->dimensions, data_cursor,
                data_cursor + tensor->payload_size) != 0) {
            goto no_memory;
        }
        if (tensor->payload_size > UINT64_MAX - data_cursor) {
            set_error("shard tensor data offset overflows");
            goto fail;
        }
        data_cursor += tensor->payload_size;
    }
    if (builder_append(&header, "}") != 0) goto no_memory;
    if (member_count == 0) {
        set_error("cannot write an empty Safetensors shard");
        goto fail;
    }
    if (header.length > UINT64_MAX - 8 - (ALIGNMENT - 1)) {
        set_error("Safetensors header size overflows");
        goto fail;
    }
    data_start = ((uint64_t)header.length + 8 + ALIGNMENT - 1) &
                 ~((uint64_t)ALIGNMENT - 1);
    header_size = data_start - 8;

    file = fopen(shard->path, "wb");
    if (file == NULL) {
        set_path_error("cannot create Safetensors shard", shard->path);
        goto fail;
    }
    for (i = 0; i < 8; ++i) encoded_length[i] = (unsigned char)(header_size >> (i * 8));
    if (write_all(file, encoded_length, sizeof(encoded_length)) != 0 ||
        write_all(file, header.data, header.length) != 0 ||
        write_space_bytes(file, header_size - header.length,
                          aligned_buffer) != 0) {
        set_path_error("cannot write Safetensors header", shard->path);
        goto fail;
    }

    data_cursor = 0;
    for (i = 0; i < model->tensor_count; ++i) {
        TensorRecord *tensor = &model->tensors[i];
        uint64_t padding;
        if (strcmp(tensor->shard_name, shard->name) != 0) continue;
        padding = tensor->shard_data_offset - data_cursor;
        if (write_zero_bytes(file, padding, aligned_buffer) != 0 ||
            write_mapped_payload(file, tensor, model, aligned_buffer) != 0) {
            set_path_error("cannot write Safetensors tensor payload",
                           shard->path);
            goto fail;
        }
        tensor->shard_offset = data_start + tensor->shard_data_offset;
        data_cursor = tensor->shard_data_offset + tensor->payload_size;
    }
    if (fflush(file) != 0 || fsync(fileno(file)) != 0) {
        set_path_error("cannot flush Safetensors shard", shard->path);
        goto fail;
    }
    {
        off_t end = ftello(file);
        if (end < 0) {
            set_path_error("cannot determine Safetensors shard size",
                           shard->path);
            goto fail;
        }
        shard->size = (uint64_t)end;
    }
    if (fclose(file) != 0) {
        file = NULL;
        set_path_error("cannot close Safetensors shard", shard->path);
        goto fail;
    }
    file = NULL;
    free(header.data);
    return 0;

no_memory:
    set_error("out of memory while building Safetensors header");
fail:
    if (file != NULL) fclose(file);
    free(header.data);
    failed = 1;
    return failed ? -1 : 0;
}

static int build_and_write_shards(ModelInfo *model, const char *weights_dir,
                                 unsigned char *aligned_buffer) {
    size_t i;

    for (i = 0; i < model->tensor_count; ++i) {
        if (get_shard(model, model->tensors[i].shard_name, weights_dir) == NULL)
            return -1;
    }
    for (i = 0; i < model->shard_count; ++i) {
        if (model->progress_fn) {
            model->progress_fn("writing", i + 1, model->shard_count,
                               model->progress_user_data);
        }
        if (write_safetensors_shard(model, i, aligned_buffer) != 0) return -1;
    }
    return 0;
}

static int append_json_string(FILE *file, const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    if (fputc('"', file) == EOF) return -1;
    while (*cursor != '\0') {
        switch (*cursor) {
        case '"': if (fputs("\\\"", file) == EOF) return -1; break;
        case '\\': if (fputs("\\\\", file) == EOF) return -1; break;
        case '\b': if (fputs("\\b", file) == EOF) return -1; break;
        case '\f': if (fputs("\\f", file) == EOF) return -1; break;
        case '\n': if (fputs("\\n", file) == EOF) return -1; break;
        case '\r': if (fputs("\\r", file) == EOF) return -1; break;
        case '\t': if (fputs("\\t", file) == EOF) return -1; break;
        default:
            if (*cursor < 0x20) {
                if (fprintf(file, "\\u%04x", (unsigned)*cursor) < 0) return -1;
            } else if (fputc(*cursor, file) == EOF) {
                return -1;
            }
        }
        ++cursor;
    }
    return fputc('"', file) == EOF ? -1 : 0;
}

static int append_shard_reference(FILE *file, const char *name) {
    StringBuilder path = {0};
    int result;
    if (builder_append(&path, ".cheetah/weights/") != 0 ||
        builder_append(&path, name) != 0) {
        free(path.data);
        return -1;
    }
    result = append_json_string(file, path.data);
    free(path.data);
    return result;
}

static char *default_chat_template(const ModelInfo *model) {
    const char *templ;
    if (strcmp(model->model_type, "llama") == 0 &&
        model->has_header_tokens && model->has_eot_id) {
        templ = "{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}";
    } else if (strcmp(model->model_type, "phi") == 0) {
        templ = "{% for message in messages %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";
    } else if (strcmp(model->model_type, "gemma") == 0) {
        templ = "{% for message in messages %}{{ '<start_of_turn>' + message['role'] + '\\n' + message['content'] + '<end_of_turn>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}";
    } else if (strcmp(model->model_type, "qwen2") == 0) {
        templ = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";
    } else {
        templ = "{% for message in messages %}{% if loop.first and message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% else %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% endif %}{% endfor %}";
    }
    return strdup(templ);
}

static void configure_stops(ModelInfo *model) {
    if (model->has_im_end) model->stop_markers[model->stop_marker_count++] = "<|im_end|>";
    if (model->has_eot_id) model->stop_markers[model->stop_marker_count++] = "<|eot_id|>";
    if (model->has_end_of_turn) model->stop_markers[model->stop_marker_count++] = "<end_of_turn>";
    if (model->has_phi_end) model->stop_markers[model->stop_marker_count++] = "<|end|>";
    if (model->has_fim_suffix) model->stop_markers[model->stop_marker_count++] = "<|fim_suffix|>";
    if (model->has_endoftext) model->stop_markers[model->stop_marker_count++] = "<|endoftext|>";
    if (model->has_end_of_text) model->stop_markers[model->stop_marker_count++] = "<|end_of_text|>";
    if (model->has_eos) model->stop_markers[model->stop_marker_count++] = "<eos>";

    if (model->stop_marker_count == 0) {
        if (strcmp(model->model_type, "phi") == 0) {
            model->stop_markers[model->stop_marker_count++] = "<|end|>";
            model->stop_markers[model->stop_marker_count++] = "<|endoftext|>";
        } else if (strcmp(model->model_type, "gemma") == 0) {
            model->stop_markers[model->stop_marker_count++] = "<end_of_turn>";
            model->stop_markers[model->stop_marker_count++] = "<eos>";
        } else if (strcmp(model->model_type, "qwen2") == 0) {
            model->stop_markers[model->stop_marker_count++] = "<|im_end|>";
            model->stop_markers[model->stop_marker_count++] = "<|endoftext|>";
        } else {
            model->stop_markers[model->stop_marker_count++] = "<|eot_id|>";
            model->stop_markers[model->stop_marker_count++] = "<|end_of_text|>";
        }
    }
}

static int write_manifest(const char *path, const ModelInfo *model) {
    FILE *file = fopen(path, "wb");
    size_t i;
    int failed = 0;

    if (file == NULL) {
        set_path_error("cannot create manifest", path);
        return -1;
    }
    failed |= fprintf(file, "{\n  \"format\": \"cheetah-vulkan-safetensors-v1\",\n"
                            "  \"alignment\": 4096,\n  \"model\": {\n"
                            "    \"model_type\": ") < 0;
    if (!failed && append_json_string(file, model->model_type) != 0) failed = 1;
    failed |= fprintf(file,
                      ",\n    \"num_hidden_layers\": %" PRIu64
                      ",\n    \"hidden_size\": %" PRIu64
                      ",\n    \"intermediate_size\": %" PRIu64
                      ",\n    \"vocab_size\": %" PRIu64
                      ",\n    \"num_attention_heads\": %" PRIu64
                      ",\n    \"num_key_value_heads\": %" PRIu64
                      ",\n    \"max_position_embeddings\": %" PRIu64,
                      model->num_layers, model->hidden_size,
                      model->intermediate_size, model->vocab_size,
                      model->num_attention_heads, model->num_key_value_heads,
                      model->max_context) < 0;
    failed |= fprintf(file,
                      ",\n    \"model_type_source\": \"%s\""
                      ",\n    \"num_attention_heads_source\": \"%s\""
                      ",\n    \"max_position_embeddings_source\": \"default (4096)\""
                      "\n  },\n  \"chat_template\": ",
                      model->model_type_from_config
                          ? "source config.json"
                          : "tensor-name heuristic",
                      model->has_q_norm
                          ? "inferred from q_norm and q_proj tensor shapes"
                          : "estimated from hidden_size; verify") < 0;
    if (!failed && append_json_string(file, model->chat_template) != 0) failed = 1;
    failed |= fprintf(file, ",\n  \"stop_markers\": [") < 0;
    for (i = 0; !failed && i < model->stop_marker_count; ++i) {
        if (i > 0 && fputs(", ", file) == EOF) failed = 1;
        if (!failed && append_json_string(file, model->stop_markers[i]) != 0)
            failed = 1;
    }
    failed |= fprintf(file, "],\n  \"shards\": [") < 0;
    for (i = 0; !failed && i < model->shard_count; ++i) {
        if (i > 0 && fputs(",", file) == EOF) failed = 1;
        if (!failed && fprintf(file, "\n    {\"group\": ") < 0) failed = 1;
        if (!failed && append_json_string(file, model->shards[i].name) != 0)
            failed = 1;
        if (!failed && fprintf(file, ", \"file\": ") < 0) failed = 1;
        if (!failed && append_shard_reference(file, model->shards[i].name) != 0)
            failed = 1;
        if (!failed && fprintf(file, ", \"size\": %" PRIu64 "}",
                               model->shards[i].size) < 0) failed = 1;
    }
    failed |= fprintf(file, "\n  ],\n  \"tensors\": [") < 0;
    for (i = 0; !failed && i < model->tensor_count; ++i) {
        const TensorRecord *tensor = &model->tensors[i];
        size_t dimension;
        if (i > 0 && fputs(",", file) == EOF) failed = 1;
        if (!failed && fputs("\n    {\"name\": ", file) == EOF) failed = 1;
        if (!failed && append_json_string(file, tensor->name) != 0) failed = 1;
        if (!failed && fputs(", \"dtype\": ", file) == EOF) failed = 1;
        if (!failed && append_json_string(file, tensor->dtype) != 0) failed = 1;
        if (!failed && fprintf(file, ", \"shape\": [") < 0) failed = 1;
        for (dimension = 0; !failed && dimension < tensor->dimensions;
             ++dimension) {
            if (dimension > 0 && fputs(", ", file) == EOF) failed = 1;
            if (!failed && fprintf(file, "%" PRIu64,
                                   tensor->shape[dimension]) < 0) failed = 1;
        }
        if (!failed && fputs("], \"file\": ", file) == EOF) failed = 1;
        if (!failed && append_shard_reference(file, tensor->shard_name) != 0)
            failed = 1;
        if (!failed && fprintf(file,
                               ", \"offset\": %" PRIu64
                               ", \"data_offset\": %" PRIu64
                               ", \"size\": %" PRIu64 "}",
                               tensor->shard_offset,
                               tensor->shard_data_offset,
                               tensor->payload_size) < 0) failed = 1;
    }
    failed |= fprintf(file, "\n  ]\n}\n") < 0;
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        set_path_error("failed writing manifest", path);
        return -1;
    }
    return 0;
}

static int write_units(const char *path, const ModelInfo *model) {
    FILE *file = fopen(path, "wb");
    size_t i;
    int failed = 0;
    if (file == NULL) {
        set_path_error("cannot create units manifest", path);
        return -1;
    }
    failed |= fprintf(file, "[") < 0;
    for (i = 0; !failed && i < model->tensor_count; ++i) {
        const TensorRecord *tensor = &model->tensors[i];
        int layer_id = 0;
        if (tensor->name != NULL) {
            const char *lp = strstr(tensor->name, "layers.");
            if (lp) {
                long layer_num = strtol(lp + 7, NULL, 10);
                layer_id = (int)layer_num;
            }
        }
        if (i > 0) { if (fputs(",", file) == EOF) failed = 1; }
        if (!failed) {
            failed |= fprintf(file, "\n  {\n    \"unit_id\": ") < 0;
            if (!failed && append_json_string(file, tensor->name) != 0) failed = 1;
            failed |= fprintf(file, ",\n    \"name\": ") < 0;
            if (!failed && append_json_string(file, tensor->name) != 0) failed = 1;
            failed |= fprintf(file, ",\n    \"dtype\": ") < 0;
            if (!failed && append_json_string(file, tensor->dtype) != 0) failed = 1;
            failed |= fprintf(file, ",\n    \"shape\": [") < 0;
            size_t dimension;
            for (dimension = 0; !failed && dimension < tensor->dimensions; ++dimension) {
                if (dimension > 0) { if (fputs(", ", file) == EOF) failed = 1; }
                if (!failed) {
                    failed |= fprintf(file, "%llu", (unsigned long long)tensor->shape[dimension]) < 0;
                }
            }
            failed |= fprintf(file, "],\n    \"bytes\": %llu,\n    \"source_file\": ",
                              (unsigned long long)tensor->payload_size) < 0;
            if (!failed && append_json_string(file, tensor->shard_name) != 0) failed = 1;
            failed |= fprintf(file, ",\n    \"source_offset\": %llu,\n    \"shard\": ",
                              (unsigned long long)tensor->shard_data_offset) < 0;
            if (!failed && append_json_string(file, tensor->shard_name) != 0) failed = 1;
            failed |= fprintf(file, ",\n    \"layer_id\": %d\n  }", layer_id) < 0;
        }
    }
    failed |= fprintf(file, "\n]\n") < 0;
    if (fclose(file) != 0) failed = 1;
    if (failed) { set_path_error("failed writing units manifest", path); return -1; }
    return 0;
}

static int copy_file(const char *source, const char *destination) {
    FILE *input = NULL;
    FILE *output = NULL;
    void *buffer = NULL;
    int result = -1;

    input = fopen(source, "rb");
    if (input == NULL) {
        set_path_error("cannot open source config", source);
        goto done;
    }
    output = fopen(destination, "wb");
    if (output == NULL) {
        set_path_error("cannot create destination config", destination);
        goto done;
    }
    if (posix_memalign(&buffer, ALIGNMENT, COPY_CHUNK) != 0) {
        set_error("posix_memalign failed while copying config");
        goto done;
    }
    for (;;) {
        size_t bytes = fread(buffer, 1, COPY_CHUNK, input);
        if (bytes > 0 && fwrite(buffer, 1, bytes, output) != bytes) {
            set_path_error("cannot write destination config", destination);
            goto done;
        }
        if (bytes < COPY_CHUNK) {
            if (ferror(input)) {
                set_path_error("cannot read source config", source);
                goto done;
            }
            break;
        }
    }
    result = 0;
done:
    free(buffer);
    if (input != NULL) fclose(input);
    if (output != NULL && fclose(output) != 0) result = -1;
    return result;
}

static int write_inferred_config(const char *path, const ModelInfo *model) {
    FILE *file = fopen(path, "wb");
    int failed;

    if (file == NULL) {
        set_path_error("cannot create inferred config", path);
        return -1;
    }
    failed = fprintf(file,
        "{\n"
        "  \"model_type\": \"%s\",\n"
        "  \"architectures\": [\"%sForCausalLM\"],\n"
        "  \"vocab_size\": %" PRIu64 ",\n"
        "  \"hidden_size\": %" PRIu64 ",\n"
        "  \"intermediate_size\": %" PRIu64 ",\n"
        "  \"num_hidden_layers\": %" PRIu64 ",\n"
        "  \"num_attention_heads\": %" PRIu64 ",\n"
        "  \"num_key_value_heads\": %" PRIu64 ",\n"
        "  \"max_position_embeddings\": %" PRIu64 ",\n"
        "  \"rms_norm_eps\": 0.00001,\n"
        "  \"rope_theta\": 10000.0,\n"
        "  \"tie_word_embeddings\": false\n"
        "}\n",
        model->model_type,
        strcmp(model->model_type, "phi") == 0 ? "Phi" :
        strcmp(model->model_type, "gemma") == 0 ? "Gemma" :
        strcmp(model->model_type, "qwen2") == 0 ? "Qwen2" : "Llama",
        model->vocab_size, model->hidden_size, model->intermediate_size,
        model->num_layers, model->num_attention_heads,
        model->num_key_value_heads, model->max_context) < 0;
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        set_path_error("cannot finish inferred config", path);
        return -1;
    }
    if (!model->found_embedding || !model->found_layer ||
        !model->found_intermediate) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_YELLOW,
                        "cheetah-init: warning: generated config has missing "
                        "or heuristic dimensions (embedding=%d layers=%d "
                        "intermediate=%d); verify it before inference\n",
                        model->found_embedding, model->found_layer,
                        model->found_intermediate);
    } else if (!model->has_q_norm && model->found_attention) {
        CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_YELLOW,
                        "cheetah-init: warning: attention-head count was "
                        "estimated from tensor dimensions; verify config.json\n");
    }
    return 0;
}

static int ensure_config(const char *source, const char *destination,
                         const ModelInfo *model) {
    struct stat status;
    if (stat(destination, &status) == 0) return 0;
    if (errno != ENOENT) {
        set_path_error("cannot stat destination config", destination);
        return -1;
    }
    if (source != NULL && stat(source, &status) == 0 && S_ISREG(status.st_mode)) {
        return copy_file(source, destination);
    }
    return write_inferred_config(destination, model);
}

static char *json_string_field_from_file(const char *path, const char *field) {
    FILE *file;
    long file_size;
    char *contents;
    char needle[128];
    char *cursor;
    JsonParser parser;
    char *value = NULL;

    if (snprintf(needle, sizeof(needle), "\"%s\"", field) < 0 ||
        strlen(needle) >= sizeof(needle)) {
        return NULL;
    }
    file = fopen(path, "rb");
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (file_size = ftell(file)) < 0 ||
        (uintmax_t)file_size > SIZE_MAX - 1 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    contents = (char *)malloc((size_t)file_size + 1);
    if (contents == NULL) {
        fclose(file);
        return NULL;
    }
    if (fread(contents, 1, (size_t)file_size, file) != (size_t)file_size) {
        free(contents);
        fclose(file);
        return NULL;
    }
    fclose(file);
    contents[file_size] = '\0';
    cursor = strstr(contents, needle);
    if (cursor != NULL) {
        cursor += strlen(needle);
        parser.cursor = cursor;
        parser.end = contents + file_size;
        skip_space(&parser);
        if (parser.cursor < parser.end && *parser.cursor++ == ':') {
            value = parse_json_string(&parser);
        }
    }
    free(contents);
    return value;
}

static const char *find_bytes(const char *begin, const char *end,
                              const char *needle, size_t needle_length) {
    const char *cursor;
    if ((size_t)(end - begin) < needle_length) return NULL;
    for (cursor = begin; cursor + needle_length <= end; ++cursor) {
        if (memcmp(cursor, needle, needle_length) == 0) return cursor;
    }
    return NULL;
}

static int special_token_present(const char *json, size_t length,
                                 const char *token) {
    char content_key[] = "\"content\"";
    const char *cursor = json;
    const char *end = json + length;
    while (cursor < end) {
        const char *key = find_bytes(cursor, end, content_key,
                                     sizeof(content_key) - 1);
        const char *object_start;
        const char *object_end;
        JsonParser parser;
        char *content;
        const char *special;
        if (key == NULL || key >= end) return 0;
        parser.cursor = key + sizeof(content_key) - 1;
        parser.end = end;
        skip_space(&parser);
        if (parser.cursor >= end || *parser.cursor++ != ':') {
            cursor = key + sizeof(content_key) - 1;
            continue;
        }
        content = parse_json_string(&parser);
        if (content == NULL) {
            cursor = key + sizeof(content_key) - 1;
            continue;
        }
        object_start = key;
        while (object_start > json && *object_start != '{') --object_start;
        object_end = parser.cursor;
        while (object_end < end && *object_end != '}') ++object_end;
        special = object_start;
        if (content[0] != '\0' && strcmp(content, token) == 0 &&
            object_start < key && object_end < end) {
            const char *special_key = "\"special\"";
            const char *scan = object_start;
            while (scan < object_end) {
                special = find_bytes(scan, object_end, special_key,
                                     strlen(special_key));
                if (special == NULL || special >= object_end) break;
                {
                    JsonParser boolean_parser;
                    boolean_parser.cursor = special + strlen(special_key);
                    boolean_parser.end = object_end;
                    skip_space(&boolean_parser);
                    if (boolean_parser.cursor < object_end &&
                        *boolean_parser.cursor++ == ':') {
                        skip_space(&boolean_parser);
                        if ((size_t)(object_end - boolean_parser.cursor) >= 4 &&
                            memcmp(boolean_parser.cursor, "true", 4) == 0) {
                            free(content);
                            return 1;
                        }
                    }
                }
                scan = special + strlen(special_key);
            }
        }
        free(content);
        cursor = parser.cursor;
    }
    return 0;
}

static int scan_special_tokens(const char *path, ModelInfo *model) {
    static const char *const candidates[] = {
        "<|im_start|>", "<|im_end|>", "<|start_header_id|>",
        "<|eot_id|>", "<|end_of_text|>", "<|endoftext|>",
        "<start_of_turn>", "<end_of_turn>", "<|end|>",
        "<|fim_suffix|>", "<eos>"
    };
    int fd;
    struct stat status;
    char *mapping;
    size_t length;
    size_t i;
    int found_any = 0;

    fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (errno == ENOENT) return 0;
        set_path_error("cannot open tokenizer file", path);
        return -1;
    }
    if (fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
        status.st_size <= 0 || (uintmax_t)status.st_size > SIZE_MAX) {
        set_path_error("invalid tokenizer file", path);
        close(fd);
        return -1;
    }
    length = (size_t)status.st_size;
    mapping = (char *)mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        set_path_error("cannot mmap tokenizer file", path);
        close(fd);
        return -1;
    }
    for (i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (!special_token_present(mapping, length, candidates[i])) continue;
        found_any = 1;
        if (strcmp(candidates[i], "<|im_start|>") == 0) model->has_im_start = 1;
        else if (strcmp(candidates[i], "<|im_end|>") == 0) model->has_im_end = 1;
        else if (strcmp(candidates[i], "<|start_header_id|>") == 0) model->has_header_tokens = 1;
        else if (strcmp(candidates[i], "<|eot_id|>") == 0) model->has_eot_id = 1;
        else if (strcmp(candidates[i], "<|end_of_text|>") == 0) model->has_end_of_text = 1;
        else if (strcmp(candidates[i], "<|endoftext|>") == 0) model->has_endoftext = 1;
        else if (strcmp(candidates[i], "<start_of_turn>") == 0) model->has_start_of_turn = 1;
        else if (strcmp(candidates[i], "<end_of_turn>") == 0) model->has_end_of_turn = 1;
        else if (strcmp(candidates[i], "<|end|>") == 0) model->has_phi_end = 1;
        else if (strcmp(candidates[i], "<|fim_suffix|>") == 0) model->has_fim_suffix = 1;
        else if (strcmp(candidates[i], "<eos>") == 0) model->has_eos = 1;
    }
    munmap(mapping, length);
    close(fd);

    if (!model->model_type_from_config) {
        if (model->has_im_start && model->has_im_end) {
            (void)snprintf(model->model_type, sizeof(model->model_type), "qwen2");
        } else if (model->has_header_tokens && model->has_eot_id) {
            (void)snprintf(model->model_type, sizeof(model->model_type), "llama");
        } else if (model->has_start_of_turn && model->has_end_of_turn) {
            (void)snprintf(model->model_type, sizeof(model->model_type), "gemma");
        } else if (model->has_phi_end) {
            (void)snprintf(model->model_type, sizeof(model->model_type), "phi");
        }
    }
    return found_any;
}

static char *model_source_companion(const char *source, int source_is_file,
                                    const char *filename) {
    char *directory = NULL;
    char *path = NULL;

    if (!source_is_file) {
        if (path_join(source, filename, &path) != 0) return NULL;
        return path;
    }
    {
        const char *slash = strrchr(source, '/');
        if (slash == NULL) {
            if (path_join(".", filename, &path) != 0) return NULL;
            return path;
        } else if (slash == source) {
            if (path_join("/", filename, &path) != 0) return NULL;
            return path;
        } else {
            size_t length = (size_t)(slash - source);
            directory = (char *)malloc(length + 1);
            if (directory != NULL) {
                memcpy(directory, source, length);
                directory[length] = '\0';
            }
        }
    }
    if (directory == NULL || path_join(directory, filename, &path) != 0) {
        free(directory);
        return NULL;
    }
    free(directory);
    return path;
}

static int remove_unsharded_weights(const char *directory) {
    DIR *dir = opendir(directory);
    struct dirent *entry;
    int result = 0;
    if (dir == NULL) {
        set_path_error("cannot open destination directory", directory);
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        char *path;
        struct stat status;
        if (strcmp(entry->d_name, ".") == 0 ||
            strcmp(entry->d_name, "..") == 0 ||
            strcmp(entry->d_name, ".cheetah") == 0) {
            continue;
        }
        if (path_join(directory, entry->d_name, &path) != 0) {
            set_error("out of memory while cleaning source weight copies");
            result = -1;
            break;
        }
        if (lstat(path, &status) == 0) {
            if (S_ISDIR(status.st_mode)) {
                if (remove_unsharded_weights(path) != 0) result = -1;
            } else if (S_ISREG(status.st_mode) &&
                       ends_with_safetensors(entry->d_name) &&
                       unlink(path) != 0) {
                set_path_error("cannot remove unsharded destination weights",
                               path);
                result = -1;
            }
        }
        free(path);
        if (result != 0) break;
    }
    closedir(dir);
    return result;
}

static int remove_stale_shards(const char *weights_dir,
                               const ModelInfo *model) {
    DIR *dir = opendir(weights_dir);
    struct dirent *entry;
    if (dir == NULL) {
        set_path_error("cannot open destination weights directory", weights_dir);
        return -1;
    }
    while ((entry = readdir(dir)) != NULL) {
        size_t i;
        char *path;
        struct stat status;
        int current = 0;
        {
            size_t name_length = strlen(entry->d_name);
            if (!ends_with_safetensors(entry->d_name) &&
                !(name_length >= 4 &&
                  strcmp(entry->d_name + name_length - 4, ".bin") == 0)) {
                continue;
            }
        }
        for (i = 0; i < model->shard_count; ++i) {
            if (strcmp(entry->d_name, model->shards[i].name) == 0) {
                current = 1;
                break;
            }
        }
        if (current) continue;
        if (path_join(weights_dir, entry->d_name, &path) != 0) {
            closedir(dir);
            set_error("out of memory while cleaning obsolete shards");
            return -1;
        }
        if (stat(path, &status) == 0 && S_ISREG(status.st_mode) &&
            unlink(path) != 0) {
            set_path_error("cannot remove obsolete shard", path);
            free(path);
            closedir(dir);
            return -1;
        }
        free(path);
    }
    closedir(dir);
    return 0;
}

static void free_model_info(ModelInfo *model) {
    size_t i;
    for (i = 0; i < model->tensor_count; ++i) {
        free(model->tensors[i].name);
        free(model->tensors[i].dtype);
        free(model->tensors[i].shape);
        free(model->tensors[i].shard_name);
    }
    for (i = 0; i < model->shard_count; ++i) {
        free(model->shards[i].name);
        free(model->shards[i].path);
    }
    for (i = 0; i < model->source_count; ++i) {
        if (model->sources[i].mapping != NULL) {
            munmap(model->sources[i].mapping, model->sources[i].mapping_size);
        }
        free(model->sources[i].path);
    }
    free(model->tensors);
    free(model->tensor_name_slots);
    free(model->shards);
    free(model->sources);
    free(model->chat_template);
}

int cheetah_init_model(const char *src_dir, const char *dst_dir) {
    return cheetah_init_model_with_progress(src_dir, dst_dir, NULL, NULL);
}

int cheetah_init_model_with_progress(const char *src_dir, const char *dst_dir,
                                     cheetah_progress_fn progress_fn,
                                     void *user_data) {
    char **source_files = NULL;
    size_t source_count = 0;
    size_t source_index;
    char *cheetah_dir = NULL;
    char *weights_dir = NULL;
    char *manifest_path = NULL;
    char *units_path = NULL;
    char *config_path = NULL;
    char *source_config = NULL;
    char *source_tokenizer_config = NULL;
    char *source_tokenizer = NULL;
    char *destination_tokenizer_config = NULL;
    char *destination_tokenizer = NULL;
    char *configured_model_type = NULL;
    char *source_chat_template = NULL;
    void *aligned_buffer = NULL;
    ModelInfo model = {0};
    int result = -1;

    model.progress_fn = progress_fn;
    model.progress_user_data = user_data;

    if (src_dir == NULL || dst_dir == NULL ||
        src_dir[0] == '\0' || dst_dir[0] == '\0') {
        set_error("src_dir and dst_dir must be non-empty paths");
        return -1;
    }
    if (collect_source_files(src_dir, &source_files, &source_count) != 0) {
        return -1;
    }
    {
        struct stat source_status;
        int source_is_file = stat(src_dir, &source_status) == 0 &&
                             S_ISREG(source_status.st_mode);
        source_config = model_source_companion(src_dir, source_is_file,
                                               "config.json");
        source_tokenizer_config = model_source_companion(
            src_dir, source_is_file, "tokenizer_config.json");
        source_tokenizer = model_source_companion(src_dir, source_is_file,
                                                  "tokenizer.json");
        if (source_config != NULL) {
            configured_model_type =
                json_string_field_from_file(source_config, "model_type");
            if (configured_model_type != NULL &&
                configured_model_type[0] != '\0' &&
                strlen(configured_model_type) < sizeof(model.model_type)) {
                (void)snprintf(model.model_type, sizeof(model.model_type), "%s",
                               configured_model_type);
                model.model_type_from_config = 1;
            }
        }
        if (source_tokenizer_config != NULL) {
            source_chat_template =
                json_string_field_from_file(source_tokenizer_config,
                                            "chat_template");
        }
        if (source_chat_template == NULL && source_config != NULL) {
            source_chat_template =
                json_string_field_from_file(source_config, "chat_template");
        }
        if (source_tokenizer != NULL &&
            scan_special_tokens(source_tokenizer, &model) < 0) {
            goto done;
        }
        if (source_tokenizer_config != NULL &&
            scan_special_tokens(source_tokenizer_config, &model) < 0) {
            goto done;
        }
    }
    if (path_join(dst_dir, ".cheetah", &cheetah_dir) != 0 ||
        path_join(cheetah_dir, "weights", &weights_dir) != 0 ||
        path_join(dst_dir, "cheetah_manifest.json", &manifest_path) != 0 ||
        path_join(dst_dir, "cheetah_units.json", &units_path) != 0 ||
        path_join(dst_dir, "config.json", &config_path) != 0 ||
        path_join(dst_dir, "tokenizer_config.json",
                  &destination_tokenizer_config) != 0 ||
        path_join(dst_dir, "tokenizer.json", &destination_tokenizer) != 0) {
        set_error("out of memory while building output paths");
        goto done;
    }
    if (make_directories(weights_dir) != 0) goto done;
    if (posix_memalign(&aligned_buffer, ALIGNMENT, COPY_CHUNK) != 0) {
        set_error("posix_memalign failed for 4 KiB-aligned transfer buffer");
        goto done;
    }
    for (source_index = 0; source_index < source_count; ++source_index) {
        if (model.progress_fn) {
            model.progress_fn("parsing", source_index + 1, source_count,
                              model.progress_user_data);
        }
        if (process_safetensors_file(source_files[source_index], &model) != 0) {
            goto done;
        }
    }
    finalize_inference(&model);
    model.chat_template = source_chat_template;
    source_chat_template = NULL;
    if (model.chat_template == NULL) {
        model.chat_template = default_chat_template(&model);
    }
    configure_stops(&model);
    if (model.chat_template == NULL) {
        set_error("out of memory while preparing chat template");
        goto done;
    }
    if (model.progress_fn) {
        model.progress_fn("preparing", 1, 1, model.progress_user_data);
    }
    if (build_and_write_shards(&model, weights_dir,
                               (unsigned char *)aligned_buffer) != 0) goto done;
    if (write_manifest(manifest_path, &model) != 0) goto done;
    if (write_units(units_path, &model) != 0) goto done;
    if (ensure_config(source_config, config_path, &model) != 0 ||
        ensure_copy(source_tokenizer, destination_tokenizer) != 0 ||
        ensure_copy(source_tokenizer_config,
                    destination_tokenizer_config) != 0) {
        goto done;
    }
    if (remove_stale_shards(weights_dir, &model) != 0 ||
        remove_unsharded_weights(dst_dir) != 0) {
        goto done;
    }
    {
        char *legacy_manifest = NULL;
        if (path_join(cheetah_dir, "cheetah_manifest.json",
                      &legacy_manifest) == 0) {
            if (unlink(legacy_manifest) != 0 && errno != ENOENT) {
                set_path_error("cannot remove obsolete manifest",
                               legacy_manifest);
                free(legacy_manifest);
                goto done;
            }
            free(legacy_manifest);
        }
    }
    if (model.progress_fn) {
        model.progress_fn("complete", 1, 1, model.progress_user_data);
    }
    result = 0;
    CHEETAH_FPRINTF(stderr, CHEETAH_COLOR_GREEN,
                    "cheetah-init: prepared %zu tensors in %zu aligned "
                    "Safetensors shards "
                    "(layers=%" PRIu64 ", hidden_size=%" PRIu64 ")\n",
                    model.tensor_count, model.shard_count, model.num_layers,
                    model.hidden_size);
done:
    free(aligned_buffer);
    free_model_info(&model);
    for (source_index = 0; source_index < source_count; ++source_index) {
        free(source_files[source_index]);
    }
    free(source_files);
    free(cheetah_dir);
    free(weights_dir);
    free(manifest_path);
    free(units_path);
    free(config_path);
    free(destination_tokenizer_config);
    free(destination_tokenizer);
    free(source_config);
    free(source_tokenizer_config);
    free(source_tokenizer);
    free(configured_model_type);
    free(source_chat_template);
    return result;
}
