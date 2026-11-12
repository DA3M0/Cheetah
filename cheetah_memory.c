#define _POSIX_C_SOURCE 200809L

#include "cheetah_memory.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    CheetahAntUnit *items;
    size_t count;
    size_t capacity;
} UnitIndex;

typedef struct CheetahAnt {
    CheetahAntSettings settings;
    int loaded;
    UnitIndex units;
    char **unit_paths;
    int requires_init;
} CheetahAnt;

static char *xstrdup(const char *s) {
    if (!s) return NULL;
    char *p = (char *)malloc(strlen(s) + 1);
    if (p) strcpy(p, s);
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (!q) {
        fprintf(stderr, "cheetah-ant: out of memory\n");
        exit(1);
    }
    return q;
}

static uint64_t parse_uint64_token(const char *s, int *ok) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0' || v > UINT64_MAX) {
        if (ok) *ok = 0;
        return 0;
    }
    if (ok) *ok = 1;
    return (uint64_t)v;
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
    long len = ftell(f);
    if (len < 0) { fclose(f); return NULL; }
    if (fseek(f, 0, SEEK_SET) != 0) { fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t r = fread(buf, 1, (size_t)len, f);
    fclose(f);
    buf[r] = '\0';
    if (out_len) *out_len = r;
    return buf;
}

static void unit_free(CheetahAntUnit *u) {
    if (!u) return;
    free(u->unit_id); free(u->name); free(u->dtype);
    free(u->source_file); free(u->shard);
}

static void unit_index_free(UnitIndex *idx) {
    for (size_t i = 0; i < idx->count; ++i) unit_free(&idx->items[i]);
    free(idx->items); idx->items = NULL; idx->count = idx->capacity = 0;
}

static int unit_index_append(UnitIndex *idx, const CheetahAntUnit *src) {
    if (idx->count == idx->capacity) {
        size_t ncap = idx->capacity ? idx->capacity * 2 : 16;
        idx->items = xrealloc(idx->items, ncap * sizeof(*idx->items));
        idx->capacity = ncap;
    }
    CheetahAntUnit *dst = &idx->items[idx->count++];
    dst->unit_id = src->unit_id ? xstrdup(src->unit_id) : NULL;
    dst->name = src->name ? xstrdup(src->name) : NULL;
    dst->dtype = src->dtype ? xstrdup(src->dtype) : NULL;
    dst->source_file = src->source_file ? xstrdup(src->source_file) : NULL;
    dst->shard = src->shard ? xstrdup(src->shard) : NULL;
    dst->ndim = src->ndim;
    dst->bytes = src->bytes;
    dst->source_offset = src->source_offset;
    dst->layer_id = src->layer_id;
    dst->tier = src->tier;
    memcpy(dst->shape, src->shape, sizeof(dst->shape));
    return 0;
}

/* Minimal JSON helpers for the controlled units manifest format. */

static void skip_ws(const char *s, size_t *i, size_t n) {
    while (*i < n && (s[*i] == ' ' || s[*i] == '\t' || s[*i] == '\n' || s[*i] == '\r'))
        (*i)++;
}

/* Get value of key k in the object that starts at s[pos]=='{'. */
static char *json_get_key(const char *s, size_t *pos, size_t n, const char *key) {
    size_t i = *pos;
    skip_ws(s, &i, n);
    if (i >= n || s[i] != '{') return NULL;
    i++;
    while (i < n) {
        skip_ws(s, &i, n);
        if (i >= n) break;
        if (s[i] == '}') { i++; break; }
        if (s[i] != '"') { i++; continue; }
        size_t klen = 0;
        while (i + klen + 1 < n && s[i + klen + 1] != '"') klen++;
        if (klen > 0 && strncmp(&s[i + 1], key, klen) == 0 && klen == strlen(key)) {
            size_t colon = i + klen + 2;
            skip_ws(s, &colon, n);
            if (colon < n && s[colon] == ':') {
                size_t val = colon + 1;
                skip_ws(s, &val, n);
                if (val < n && s[val] == '"') {
                    size_t v = val + 1;
                    size_t end = v;
                    while (end < n && s[end] != '"') end++;
                    if (end < n) {
                        char *out = (char *)malloc(end - v + 1);
                        if (out) {
                            memcpy(out, &s[v], end - v);
                            out[end - v] = '\0';
                        }
                        *pos = end + 1;
                        return out;
                    }
                }
            }
        }
        size_t j = i;
        while (j < n && s[j] != ',' && s[j] != '}') j++;
        i = j;
    }
    return NULL;
}

static char *json_object_get(const char *s, const char *key) {
    size_t n = strlen(s);
    size_t pos = 0;
    return json_get_key(s, &pos, n, key);
}

/* Read and parse cheetah_units.json from dst_dir into idx. */
static int load_units_from_json(CheetahAnt *ant, const char *dst_dir, UnitIndex *idx) {
    (void)ant;
    char path[4096];
    snprintf(path, sizeof(path), "%s/cheetah_units.json", dst_dir);
    char *text = read_file(path, NULL);
    if (!text) {
        fprintf(stderr,
                "cheetah-ant: %s/cheetah_units.json not found; run 'cheetah init'.\n",
                dst_dir);
        return -1;
    }
    size_t n = strlen(text);
    size_t i = 0;
    skip_ws(text, &i, n);
    if (i >= n || text[i] != '[') {
        fprintf(stderr, "cheetah-ant: %s/cheetah_units.json is not an array.\n", dst_dir);
        free(text);
        return -1;
    }
    i++;
    while (1) {
        skip_ws(text, &i, n);
        if (i >= n) break;
        if (text[i] == ']') { i++; break; }
        if (text[i] == ',') { i++; continue; }
        size_t obj_start = i;
        size_t depth = 0;
        while (i < n) {
            if (text[i] == '{') depth++;
            else if (text[i] == '}') { depth--; if (depth == 0) break; }
            i++;
        }
        if (i >= n) { free(text); return -1; }
        i++; /* move past closing brace */
        char *obj = xstrdup(&text[obj_start]);
        CheetahAntUnit u;
        memset(&u, 0, sizeof(u));
        u.unit_id = json_object_get(obj, "unit_id");
        u.name = json_object_get(obj, "name");
        u.dtype = json_object_get(obj, "dtype");
        u.source_file = json_object_get(obj, "source_file");
        u.shard = json_object_get(obj, "shard");
        char *shape_str = json_object_get(obj, "shape");
        if (!shape_str) shape_str = json_object_get(obj, "dims");
        if (shape_str) {
            size_t j = 0;
            char *save = NULL;
            char *tok = strtok_r(shape_str, " [],", &save);
            while (tok && j < 8) {
                int ok = 0;
                u.shape[j] = parse_uint64_token(tok, &ok);
                if (!ok) u.shape[j] = 0;
                j++;
                tok = strtok_r(NULL, " [],", &save);
            }
            u.ndim = j;
        }
        char *bytes_str = json_object_get(obj, "bytes");
        if (bytes_str) u.bytes = parse_uint64_token(bytes_str, NULL);
        char *layer_str = json_object_get(obj, "layer_id");
        if (layer_str) u.layer_id = (int)parse_uint64_token(layer_str, NULL);
        char *offset_str = json_object_get(obj, "source_offset");
        if (offset_str) u.source_offset = parse_uint64_token(offset_str, NULL);
        free(obj);
        free(shape_str);
        free(bytes_str);
        free(layer_str);
        free(offset_str);
        if (unit_index_append(idx, &u) != 0) {
            free(text);
            return -1;
        }
        free(u.unit_id); free(u.name); free(u.dtype);
        free(u.source_file); free(u.shard);
        skip_ws(text, &i, n);
        if (i < n && text[i] == ',') i++;
    }
    free(text);
    return 0;
}

int cheetah_ant_unit_list(const CheetahAnt *ant, const char *dst_dir,
                          CheetahAntUnit **units_out, size_t *count_out) {
    (void)dst_dir;
    if (!ant->loaded) return -1;
    UnitIndex *idx = (UnitIndex *)&ant->units;
    if (idx->count == 0) return -1;
    CheetahAntUnit *arr = (CheetahAntUnit *)malloc(idx->count * sizeof(*arr));
    if (!arr) return -1;
    for (size_t i = 0; i < idx->count; ++i) {
        arr[i] = idx->items[i];
        arr[i].unit_id = xstrdup(arr[i].unit_id);
        arr[i].name = xstrdup(arr[i].name);
        arr[i].dtype = xstrdup(arr[i].dtype);
        arr[i].source_file = xstrdup(arr[i].source_file);
        arr[i].shard = xstrdup(arr[i].shard);
    }
    *units_out = arr;
    *count_out = idx->count;
    return 0;
}

void cheetah_ant_unit_list_free(CheetahAnt *ant, CheetahAntUnit *units, size_t count) {
    (void)ant;
    if (!units) return;
    for (size_t i = 0; i < count; ++i) unit_free(&units[i]);
    free(units);
}

int cheetah_ant_set_settings(CheetahAnt *ant, const CheetahAntSettings *s) {
    if (!ant) return -1;
    ant->settings = *s;
    return 0;
}

int cheetah_ant_load_model(CheetahAnt *ant, const char *dst_dir) {
    if (!ant) return -1;
    if (load_units_from_json(ant, dst_dir, &ant->units) != 0) return -1;
    ant->loaded = 1;
    return 0;
}

static CheetahAntUnit *unit_index_find(UnitIndex *idx, const char *unit_id) {
    if (!idx) return NULL;
    for (size_t i = 0; i < idx->count; ++i) {
        if (idx->items[i].unit_id && strcmp(idx->items[i].unit_id, unit_id) == 0)
            return &idx->items[i];
    }
    return NULL;
}

int cheetah_ant_unit_tier(const CheetahAnt *ant, const char *dst_dir,
                          const char *unit_id, int *tier_out) {
    (void)dst_dir;
    if (!ant || !unit_id || !tier_out) return -1;
    CheetahAntUnit *u = unit_index_find((UnitIndex *)&ant->units, unit_id);
    if (!u) return -1;
    *tier_out = u->tier;
    return 0;
}

int cheetah_ant_unit_pin(const CheetahAnt *ant, const char *unit_id, int pin) {
    if (!ant || !unit_id) return -1;
    CheetahAntUnit *u = unit_index_find((UnitIndex *)&ant->units, unit_id);
    if (!u) return -1;
    u->tier = pin ? CHEETAH_ANT_TIER_RAM : CHEETAH_ANT_TIER_DISK;
    return 0;
}

int cheetah_ant_evict_unit(CheetahAnt *ant, const char *dst_dir, const char *unit_id) {
    (void)dst_dir;
    if (!ant || !unit_id) return -1;
    CheetahAntUnit *u = unit_index_find(&ant->units, unit_id);
    if (!u) return -1;
    u->tier = CHEETAH_ANT_TIER_DISK;
    return 0;
}

int cheetah_ant_page_in(CheetahAnt *ant, const char *unit_id, CheetahAntTier target_tier) {
    if (!ant || !unit_id) return -1;
    CheetahAntUnit *u = unit_index_find(&ant->units, unit_id);
    if (!u) return -1;
    u->tier = target_tier;
    return 0;
}

int cheetah_ant_model_is_initialized(const CheetahAnt *ant, const char *dst_dir) {
    if (!ant || !dst_dir) return 0;
    char path[4096];
    snprintf(path, sizeof(path), "%s/.cheetah/weights", dst_dir);
    return access(path, F_OK) == 0;
}

int cheetah_ant_requires_init(const CheetahAnt *ant) {
    return ant ? ant->requires_init : 1;
}

void cheetah_ant_model_not_initialized_warning(const char *model_id, const char *dst_dir) {
    if (!model_id) return;
    fprintf(stderr,
            "cheetah: warning: model '%s' is not initialized for Ant memory optimization.\n",
            model_id);
    if (dst_dir) fprintf(stderr, "cheetah: warning:   run `cheetah init` first.\n");
}

CheetahAnt *cheetah_ant_new(void) {
    CheetahAnt *ant = (CheetahAnt *)calloc(1, sizeof(*ant));
    if (!ant) return NULL;
    ant->requires_init = 1;
    return ant;
}

void cheetah_ant_delete(CheetahAnt *ant) {
    if (!ant) return;
    unit_index_free(&ant->units);
    free(ant);
}
