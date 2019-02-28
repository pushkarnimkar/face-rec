#define DEVELOPMENT_ENV
#ifdef DEVELOPMENT_ENV
# define LOG(msg) {printf(msg);}
#else 
# define LOG(msg) {}
#endif