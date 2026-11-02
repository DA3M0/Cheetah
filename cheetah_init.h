#ifndef CHEETAH_INIT_H
#define CHEETAH_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

/* Prepare Safetensors weights for aligned, mmap/Vulkan-oriented loading.
 * src_dir may be a directory containing .safetensors files (including nested
 * directories) or one such file. Layer/expert shards are written beneath
 * dst_dir/.cheetah/weights, with cheetah_manifest.json and config.json at
 * dst_dir. Returns 0 on success, -1 on failure. */
int cheetah_init_model(const char *src_dir, const char *dst_dir);

#ifdef __cplusplus
}
#endif

#endif
