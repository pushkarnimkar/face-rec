#ifndef UTILS_H
#define UTILS_H

/** 
 * Argument is pointer to base of two byte size value.
 * This reads size and seeks the pointer inp to next location
 */
#define READSIZE2B(inp) ((*(inp++) << 8) + *(inp++))
#define MAX(v0,v1) (v0 > v1 ? v0 : v1)
#define MIN(v0,v1) (v0 < v1 ? v0 : v1)

#endif // ifndef UTILS_H