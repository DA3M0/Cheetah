#ifndef CHEETAH_MEMORY_H
#define CHEETAH_MEMORY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Where a resident unit currently lives. */
typedef enum {
    CHEETAH_ANT_TIER_VRAM,
    CHEETAH_ANT_TIER_RAM,
    CHEETAH_ANT_TIER_DISK,
} CheetahAntTier;

/* Eviction class / liveness of a resident unit. */
typedef enum {
    CHEETAH_ANT_RESIDENT,
    CHEETAH_ANT_HOT,
    CHEETAH_ANT_WARM,
    CHEETAH_ANT_COLD,
} CheetahAntResidency;

/* One atomic per-tensor unit produced by `cheetah init`. */
typedef struct {
    char *unit_id;
    char *name;
    char *dtype;
    uint64_t shape[8];
    size_t ndim;
    uint64_t bytes;
    char *source_file;
    uint64_t source_offset;
    char *shard;
    int layer_id;
    int tier;
} CheetahAntUnit;

/* Lifetime of a loaded unit (cache policy counters). */
typedef struct {
    char *unit_id;
    uint64_t ref_count;
    uint64_t last_access;
} CheetahAntLifetime;

typedef struct CheetahAnt CheetahAnt;

/* Lifecycle. */
CheetahAnt *cheetah_ant_new(void);
void cheetah_ant_delete(CheetahAnt *ant);

/* User-side memory settings (model_setting), applied before load. */
typedef struct {
    int enabled;
    uint64_t vram_pool_bytes;
    uint64_t ram_budget_bytes;
    char *disk_dir;
    char **resident_units;
    size_t resident_units_count;
    char *eviction_policy;
    int prefetch_stages;
    int allow_vram_swap;
} CheetahAntSettings;

int cheetah_ant_set_settings(CheetahAnt *ant, const CheetahAntSettings *settings);

/* Load a prepared model into memory manager. */
int cheetah_ant_load_model(CheetahAnt *ant, const char *dst_dir);

/* Unit enumeration (unit files are read via our index, not by re-parsing). */
int cheetah_ant_unit_list(const CheetahAnt *ant, const char *dst_dir,
                          CheetahAntUnit **units_out, size_t *count_out);
void cheetah_ant_unit_list_free(CheetahAnt *ant, CheetahAntUnit *units, size_t count);

/* Tier queries and residency mutations. */
int cheetah_ant_unit_tier(const CheetahAnt *ant, const char *dst_dir,
                          const char *unit_id, int *tier_out);
int cheetah_ant_unit_pin(const CheetahAnt *ant, const char *unit_id, int pin);
int cheetah_ant_evict_unit(CheetahAnt *ant, const char *dst_dir, const char *unit_id);
int cheetah_ant_page_in(CheetahAnt *ant, const char *unit_id, CheetahAntTier target_tier);

/* Model-init state (required before Ant can manage this model). */
int cheetah_ant_model_is_initialized(const CheetahAnt *ant, const char *dst_dir);
int cheetah_ant_requires_init(const CheetahAnt *ant);
void cheetah_ant_model_not_initialized_warning(const char *model_id,
                                               const char *dst_dir);

#ifdef __cplusplus
}
#endif

#endif /* CHEETAH_MEMORY_H */
