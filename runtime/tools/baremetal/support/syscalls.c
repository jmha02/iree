// See LICENSE for license details.

#include <stdint.h>
#include <string.h>
#include <stdarg.h>
#include <stdio.h>
#include <limits.h>
#include <errno.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include "util.h"

#define SYS_write 64
#define SYS_openat 56
#define SYS_close 57

// HTIF flag constants — must match the values in
// runtime/src/trace_binary_sink.c so the openat(2) host call sees
// standard Linux O_* flags.
#define HTIF_O_WRONLY 0x0001
#define HTIF_O_CREAT  0x0040
#define HTIF_O_TRUNC  0x0200

#define IREE_VFS_FD_BASE 100
#define IREE_VFS_MAX_OPEN_FILES 16

#ifndef AT_FDCWD
#define AT_FDCWD (-100)
#endif

#undef strcmp

extern volatile uint64_t tohost;
extern volatile uint64_t fromhost;

typedef struct iree_baremetal_vfs_file_t {
  const char* path;
  const uint8_t* data;
  size_t size;
} iree_baremetal_vfs_file_t;

typedef struct iree_baremetal_vfs_open_file_t {
  const iree_baremetal_vfs_file_t* file;
  size_t offset;
} iree_baremetal_vfs_open_file_t;

__attribute__((weak)) const iree_baremetal_vfs_file_t*
iree_baremetal_vfs_files(size_t* out_count) {
  if (out_count) *out_count = 0;
  return NULL;
}

static iree_baremetal_vfs_open_file_t g_vfs_open_files[IREE_VFS_MAX_OPEN_FILES];

static int iree_baremetal_vfs_fd_index(int fd) {
  int index = fd - IREE_VFS_FD_BASE;
  return index >= 0 && index < IREE_VFS_MAX_OPEN_FILES ? index : -1;
}

static int iree_baremetal_vfs_open(const char* path) {
  size_t file_count = 0;
  const iree_baremetal_vfs_file_t* files =
      iree_baremetal_vfs_files(&file_count);
  if (!files || !path) return -1;
  for (size_t i = 0; i < file_count; ++i) {
    if (files[i].path && strcmp(files[i].path, path) == 0) {
      for (int j = 0; j < IREE_VFS_MAX_OPEN_FILES; ++j) {
        if (!g_vfs_open_files[j].file) {
          g_vfs_open_files[j].file = &files[i];
          g_vfs_open_files[j].offset = 0;
          return IREE_VFS_FD_BASE + j;
        }
      }
      errno = EMFILE;
      return -1;
    }
  }
  return -1;
}

static uintptr_t syscall(uintptr_t which, uint64_t arg0, uint64_t arg1, uint64_t arg2)
{
  volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
  magic_mem[0] = which;
  magic_mem[1] = arg0;
  magic_mem[2] = arg1;
  magic_mem[3] = arg2;
  __sync_synchronize();

  tohost = (uintptr_t)magic_mem;
  while (fromhost == 0)
    ;
  fromhost = 0;

  __sync_synchronize();
  return magic_mem[0];
}

#define NUM_COUNTERS 2
static uintptr_t counters[NUM_COUNTERS];
static char* counter_names[NUM_COUNTERS];

void setStats(int enable)
{
  int i = 0;
#define READ_CTR(name) do { \
    while (i >= NUM_COUNTERS) ; \
    uintptr_t csr = read_csr(name); \
    if (!enable) { csr -= counters[i]; counter_names[i] = #name; } \
    counters[i++] = csr; \
  } while (0)

  READ_CTR(mcycle);
  READ_CTR(minstret);

#undef READ_CTR
}

void __attribute__((noreturn)) tohost_exit(uintptr_t code)
{
  tohost = (code << 1) | 1;
  while (1);
}

uintptr_t __attribute__((weak)) handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32])
{
  printf("TRAP: cause=0x%lx epc=0x%lx mtval=0x%lx\n",
         cause, epc, read_csr(mtval));
  tohost_exit(1337);
}

void exit(int code)
{
  tohost_exit(code);
}

void abort()
{
  exit(128 + SIGABRT);
}

void printstr(const char* s)
{
  syscall(SYS_write, 1, (uintptr_t)s, strlen(s));
}

// Variant of `syscall` for fesvr calls that need more than three argument
// words (e.g. SYS_openat which takes dirfd, path, len, flags, mode).
static uintptr_t syscall5(uintptr_t which, uint64_t arg0, uint64_t arg1,
                          uint64_t arg2, uint64_t arg3, uint64_t arg4)
{
  volatile uint64_t magic_mem[8] __attribute__((aligned(64)));
  magic_mem[0] = which;
  magic_mem[1] = arg0;
  magic_mem[2] = arg1;
  magic_mem[3] = arg2;
  magic_mem[4] = arg3;
  magic_mem[5] = arg4;
  __sync_synchronize();

  tohost = (uintptr_t)magic_mem;
  while (fromhost == 0)
    ;
  fromhost = 0;

  __sync_synchronize();
  return magic_mem[0];
}

// Host-filesystem helpers used by `runtime/src/trace_binary_sink.c` to write
// `flow.tensor.trace` outputs to disk through fesvr. The implementations
// here go through the same magic-mem HTIF protocol as `syscall(SYS_write,
// ...)` already does for stdout, just with the openat/close opcodes.
int htif_open(const char* path, size_t path_len, int flags, int mode)
{
  return (int)syscall5(SYS_openat, AT_FDCWD, (uintptr_t)path, path_len, flags, mode);
}

int htif_close(int fd)
{
  return (int)syscall(SYS_close, (uint64_t)fd, 0, 0);
}

int htif_write(int fd, const void* buf, size_t count)
{
  return (int)syscall(SYS_write, (uint64_t)fd, (uintptr_t)buf, count);
}

void __attribute__((weak)) thread_entry(int cid, int nc)
{
  // multi-threaded programs override this function.
  // for the case of single-threaded programs, only let core 0 proceed.
  while (cid != 0);
}

int __attribute__((weak)) main(int argc, char** argv)
{
  // single-threaded programs override this function.
  printstr("Implement main(), foo!\n");
  return -1;
}

static void init_tls(void) {}

typedef void (*init_fn_t)(void);
extern init_fn_t __preinit_array_start[];
extern init_fn_t __preinit_array_end[];
extern init_fn_t __init_array_start[];
extern init_fn_t __init_array_end[];

static void run_static_initializers(void) {
  for (init_fn_t* fn = __preinit_array_start; fn != __preinit_array_end; ++fn) {
    (*fn)();
  }
  for (init_fn_t* fn = __init_array_start; fn != __init_array_end; ++fn) {
    (*fn)();
  }
}

void _init(int cid, int nc)
{
  init_tls();
  run_static_initializers();
  thread_entry(cid, nc);

  // only single-threaded programs should ever get here.
  int ret = main(0, 0);

  char buf[NUM_COUNTERS * 32] __attribute__((aligned(64)));
  char* pbuf = buf;
  for (int i = 0; i < NUM_COUNTERS; i++)
    if (counters[i])
      pbuf += sprintf(pbuf, "%s = %d\n", counter_names[i], counters[i]);
  if (pbuf != buf)
    printstr(buf);

  exit(ret);
}

#undef putchar
int putchar(int ch)
{
  static char buf[64] __attribute__((aligned(64)));
  static int buflen = 0;

  buf[buflen++] = ch;

  if (ch == '\n' || buflen == sizeof(buf))
  {
    syscall(SYS_write, 1, (uintptr_t)buf, buflen);
    buflen = 0;
  }

  return 0;
}

void printhex(uint64_t x)
{
  char str[17];
  int i;
  for (i = 0; i < 16; i++)
  {
    str[15-i] = (x & 0xF) + ((x & 0xF) < 10 ? '0' : 'a'-10);
    x >>= 4;
  }
  str[16] = 0;

  printstr(str);
}

static char* sbrk_heap_end;
extern char _end;

void* _sbrk(ptrdiff_t incr) {
  if (!sbrk_heap_end) {
    sbrk_heap_end = &_end;
  }
  char* prev = sbrk_heap_end;
  char* next = prev + incr;
  register char* stack_ptr asm("sp");
  const ptrdiff_t kStackGuardBytes = 4096;

  if (incr > 0) {
    if (next + kStackGuardBytes > stack_ptr) {
      errno = ENOMEM;
      return (void*)-1;
    }
  } else if (incr < 0 && next < &_end) {
    next = &_end;
  }

  sbrk_heap_end = next;
  return prev;
}

int _write(int fd, const void* buf, size_t count) {
  (void)fd;
  syscall(SYS_write, 1, (uintptr_t)buf, count);
  return (int)count;
}

int _read(int fd, void* buf, size_t count) {
  int index = iree_baremetal_vfs_fd_index(fd);
  if (index < 0 || !g_vfs_open_files[index].file) {
    return 0;
  }
  const iree_baremetal_vfs_file_t* file = g_vfs_open_files[index].file;
  size_t offset = g_vfs_open_files[index].offset;
  if (offset >= file->size) return 0;
  size_t remaining = file->size - offset;
  size_t read_count = count < remaining ? count : remaining;
  memcpy(buf, file->data + offset, read_count);
  g_vfs_open_files[index].offset = offset + read_count;
  return (int)read_count;
}

int _close(int fd) {
  int index = iree_baremetal_vfs_fd_index(fd);
  if (index >= 0 && g_vfs_open_files[index].file) {
    g_vfs_open_files[index].file = NULL;
    g_vfs_open_files[index].offset = 0;
    return 0;
  }
  return htif_close(fd);
}

int _open(const char* path, int flags, int mode) {
  if (!path) {
    errno = EINVAL;
    return -1;
  }
  int vfs_fd = iree_baremetal_vfs_open(path);
  if (vfs_fd >= 0) return vfs_fd;
  return htif_open(path, strlen(path), flags, mode);
}

int _lseek(int fd, int ptr, int dir) {
  int index = iree_baremetal_vfs_fd_index(fd);
  if (index >= 0 && g_vfs_open_files[index].file) {
    const iree_baremetal_vfs_file_t* file = g_vfs_open_files[index].file;
    size_t next = 0;
    if (dir == 0) {
      next = ptr < 0 ? 0 : (size_t)ptr;
    } else if (dir == 1) {
      long long rel = (long long)g_vfs_open_files[index].offset + ptr;
      next = rel < 0 ? 0 : (size_t)rel;
    } else if (dir == 2) {
      long long rel = (long long)file->size + ptr;
      next = rel < 0 ? 0 : (size_t)rel;
    }
    if (next > file->size) next = file->size;
    g_vfs_open_files[index].offset = next;
    return (int)next;
  }
  return 0;
}

int _fstat(int fd, struct stat* st) {
  if (st) {
    memset(st, 0, sizeof(*st));
    int index = iree_baremetal_vfs_fd_index(fd);
    if (index >= 0 && g_vfs_open_files[index].file) {
      st->st_mode = S_IFREG;
      st->st_size = g_vfs_open_files[index].file->size;
    } else {
      st->st_mode = S_IFCHR;
    }
  }
  return 0;
}

int _isatty(int fd) {
  (void)fd;
  return 1;
}

static inline void printnum(void (*putch)(int, void**), void **putdat,
                    unsigned long long num, unsigned base, int width, int padc)
{
  unsigned digs[sizeof(num)*CHAR_BIT];
  int pos = 0;

  while (1)
  {
    digs[pos++] = num % base;
    if (num < base)
      break;
    num /= base;
  }

  while (width-- > pos)
    putch(padc, putdat);

  while (pos-- > 0)
    putch(digs[pos] + (digs[pos] >= 10 ? 'a' - 10 : '0'), putdat);
}

static unsigned long long getuint(va_list *ap, int lflag)
{
  if (lflag >= 2)
    return va_arg(*ap, unsigned long long);
  else if (lflag)
    return va_arg(*ap, unsigned long);
  else
    return va_arg(*ap, unsigned int);
}

static long long getint(va_list *ap, int lflag)
{
  if (lflag >= 2)
    return va_arg(*ap, long long);
  else if (lflag)
    return va_arg(*ap, long);
  else
    return va_arg(*ap, int);
}

static void vprintfmt(void (*putch)(int, void**), void **putdat, const char *fmt, va_list ap)
{
  register const char* p;
  const char* last_fmt;
  register int ch, err;
  unsigned long long num;
  int base, lflag, width, precision, altflag;
  char padc;

  while (1) {
    while ((ch = *(unsigned char *) fmt) != '%') {
      if (ch == '\0')
        return;
      fmt++;
      putch(ch, putdat);
    }
    fmt++;

    // Process a %-escape sequence
    last_fmt = fmt;
    padc = ' ';
    width = -1;
    precision = -1;
    lflag = 0;
    altflag = 0;
  reswitch:
    switch (ch = *(unsigned char *) fmt++) {

    // flag to pad on the right
    case '-':
      padc = '-';
      goto reswitch;
      
    // flag to pad with 0's instead of spaces
    case '0':
      padc = '0';
      goto reswitch;

    // width field
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      for (precision = 0; ; ++fmt) {
        precision = precision * 10 + ch - '0';
        ch = *fmt;
        if (ch < '0' || ch > '9')
          break;
      }
      goto process_precision;

    case '*':
      precision = va_arg(ap, int);
      goto process_precision;

    case '.':
      if (width < 0)
        width = 0;
      goto reswitch;

    case '#':
      altflag = 1;
      goto reswitch;

    process_precision:
      if (width < 0)
        width = precision, precision = -1;
      goto reswitch;

    // long flag (doubled for long long)
    case 'l':
      lflag++;
      goto reswitch;

    // character
    case 'c':
      putch(va_arg(ap, int), putdat);
      break;

    // string
    case 's':
      if ((p = va_arg(ap, char *)) == NULL)
        p = "(null)";
      if (width > 0 && padc != '-')
        for (width -= strnlen(p, precision); width > 0; width--)
          putch(padc, putdat);
      for (; (ch = *p) != '\0' && (precision < 0 || --precision >= 0); width--) {
        putch(ch, putdat);
        p++;
      }
      for (; width > 0; width--)
        putch(' ', putdat);
      break;

    // (signed) decimal
    case 'd':
      num = getint(&ap, lflag);
      if ((long long) num < 0) {
        putch('-', putdat);
        num = -(long long) num;
      }
      base = 10;
      goto signed_number;

    // unsigned decimal
    case 'u':
      base = 10;
      goto unsigned_number;

    // (unsigned) octal
    case 'o':
      // should do something with padding so it's always 3 octits
      base = 8;
      goto unsigned_number;

    // pointer
    case 'p':
      static_assert(sizeof(long) == sizeof(void*));
      lflag = 1;
      putch('0', putdat);
      putch('x', putdat);
      /* fall through to 'x' */

    // (unsigned) hexadecimal
    case 'x':
      base = 16;
    unsigned_number:
      num = getuint(&ap, lflag);
    signed_number:
      printnum(putch, putdat, num, base, width, padc);
      break;

    // escaped '%' character
    case '%':
      putch(ch, putdat);
      break;
      
    // unrecognized escape sequence - just print it literally
    default:
      putch('%', putdat);
      fmt = last_fmt;
      break;
    }
  }
}

int printf(const char* fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);

  vprintfmt((void*)putchar, 0, fmt, ap);

  va_end(ap);
  return 0; // incorrect return value, but who cares, anyway?
}

int sprintf(char* str, const char* fmt, ...)
{
  va_list ap;
  char* str0 = str;
  va_start(ap, fmt);

  void sprintf_putch(int ch, void** data)
  {
    char** pstr = (char**)data;
    **pstr = ch;
    (*pstr)++;
  }

  vprintfmt(sprintf_putch, (void**)&str, fmt, ap);
  *str = 0;

  va_end(ap);
  return str - str0;
}

void* memcpy(void* dest, const void* src, size_t len)
{
  if ((((uintptr_t)dest | (uintptr_t)src | len) & (sizeof(uintptr_t)-1)) == 0) {
    const uintptr_t* s = src;
    uintptr_t *d = dest;
    while (d < (uintptr_t*)(dest + len))
      *d++ = *s++;
  } else {
    const char* s = src;
    char *d = dest;
    while (d < (char*)(dest + len))
      *d++ = *s++;
  }
  return dest;
}

void* memset(void* dest, int byte, size_t len)
{
  if ((((uintptr_t)dest | len) & (sizeof(uintptr_t)-1)) == 0) {
    uintptr_t word = byte & 0xFF;
    word |= word << 8;
    word |= word << 16;
    word |= word << 16 << 16;

    uintptr_t *d = dest;
    while (d < (uintptr_t*)(dest + len))
      *d++ = word;
  } else {
    char *d = dest;
    while (d < (char*)(dest + len))
      *d++ = byte;
  }
  return dest;
}

size_t strlen(const char *s)
{
  const char *p = s;
  while (*p)
    p++;
  return p - s;
}

size_t strnlen(const char *s, size_t n)
{
  const char *p = s;
  while (n-- && *p)
    p++;
  return p - s;
}

int strcmp(const char* s1, const char* s2)
{
  unsigned char c1, c2;

  do {
    c1 = *s1++;
    c2 = *s2++;
  } while (c1 != 0 && c1 == c2);

  return c1 - c2;
}

char* strcpy(char* dest, const char* src)
{
  char* d = dest;
  while ((*d++ = *src++))
    ;
  return dest;
}

long atol(const char* str)
{
  long res = 0;
  int sign = 0;

  while (*str == ' ')
    str++;

  if (*str == '-' || *str == '+') {
    sign = *str == '-';
    str++;
  }

  while (*str) {
    res *= 10;
    res += *str++ - '0';
  }

  return sign ? -res : res;
}
