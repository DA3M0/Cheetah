#ifndef CHEETAH_INIT_H
#define CHEETAH_INIT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Prepare Safetensors weights for aligned, mmap/Vulkan-oriented loading.
 * src_dir may be a directory containing .safetensors files (including nested
 * directories) or one such file. Layer/expert shards are written beneath
 * dst_dir/.cheetah/weights, with cheetah_manifest.json and config.json at
 * dst_dir. Returns 0 on success, -1 on failure. */
int cheetah_init_model(const char *src_dir, const char *dst_dir);

/* Progress callback: phase is a human-readable step name, current/total are
 * 1-based/total counts for that phase. user_data is the opaque pointer passed
 * to cheetah_init_model_with_progress. */
typedef void (*cheetah_progress_fn)(const char *phase, size_t current,
                                    size_t total, void *user_data);

/* Same as cheetah_init_model but reports progress via callback. */
int cheetah_init_model_with_progress(const char *src_dir, const char *dst_dir,
                                     cheetah_progress_fn progress_fn,
                                     void *user_data);

#ifdef __cplusplus
}
#endif

#endif
