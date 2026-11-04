#ifndef CHEETAH_COLOR_H
#define CHEETAH_COLOR_H

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <io.h>
#define cheetah_isatty _isatty
#define cheetah_fileno _fileno
#else
#include <unistd.h>
#define cheetah_isatty isatty
#define cheetah_fileno fileno
#endif

#define CHEETAH_COLOR_CYAN "\x1b[36m"
#define CHEETAH_COLOR_GREEN "\x1b[32m"
#define CHEETAH_COLOR_YELLOW "\x1b[33m"
#define CHEETAH_COLOR_RED "\x1b[31m"
#define CHEETAH_COLOR_RESET "\x1b[0m"

static inline int cheetah_color_enabled(FILE *stream) {
    int saved_errno = errno;
    int enabled =
        getenv("NO_COLOR") == NULL && cheetah_isatty(cheetah_fileno(stream));
    errno = saved_errno;
    return enabled;
}

#define CHEETAH_FPRINTF(stream, color, ...)                                   \
    do {                                                                      \
        int cheetah_saved_errno = errno;                                      \
        int cheetah_use_color = cheetah_color_enabled(stream);                \
        if (cheetah_use_color) fputs((color), (stream));                      \
        errno = cheetah_saved_errno;                                          \
        fprintf((stream), __VA_ARGS__);                                       \
        if (cheetah_use_color) fputs(CHEETAH_COLOR_RESET, (stream));          \
    } while (0)

#endif
