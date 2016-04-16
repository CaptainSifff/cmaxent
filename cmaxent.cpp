#include <string>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <sstream>
//#include <random>// -> C++11
#include <utility>
#include <iostream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cassert>

#include <unistd.h>//POSIX specifies the presence of this header, not sure by which flag to protect it, though

#include "cmaxent.h"

const uint32_t page_size = ( size_t ) sysconf ( _SC_PAGESIZE );

/** A couple of words of explanation:
 * To get rid of some integer divisions(Nehalem: 17 - 28 cycles) we will employ bit shifts.
 * To get the power of two of the page size we employ the clz(count leading zeroes) function
 * This gets on x86 translated to the bsr instruction. It is fast on Intel Hardware if newer than core2. it's also reasonably fast on the bulldozer.
 * */
#ifdef __GNUC__
const uint32_t page_size_exponent = sizeof ( uint32_t ) *8 - 1 - __builtin_clz ( page_size );
#else
uint32_t page_size_exponent;
#endif
const size_t CLS = 64;//Cachelinesize is assumed to be 64bytes. Valid for almost all CPUs(known exceptions: AMD K7, Intel Atom, AMD E350)

#include <x86intrin.h>

using namespace std;

#define GCC_VER(x,y,z)  ((x) * 10000 + (y) * 100 + (z))

#ifndef GCC_VERSION
#define GCC_VERSION GCC_VER(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#endif

#ifdef __GNUC__
#ifndef DEBUGPRED
#define unlikely(expr) __builtin_expect(!!(expr), 0)
#define likely(expr) __builtin_expect(!!(expr), 1)
#else
asm ( ".section predict_data, \"aw\"; .previous\n"
      ".section predict_line, \"a\"; .previous\n"
      ".section predict_file, \"a\"; .previous" );
#ifdef __x86_64__
#define debugpred__(e, E) \
({ long int _e = !!(e); \
asm volatile (".pushsection predict_data\n" \
              "..predictcnt%=: .quad 0; .quad 0\n" \
              ".section predict_line; .quad %c1\n" \
              ".section predict_file; .quad %c2; .popsection\n" \
              "addq $1,..predictcnt%=(,%0,8)" \
              : : "r" ((unsigned long)(_e == E)), "i" (__LINE__), "i" (__FILE__)); \
__builtin_expect (_e, E); \
})
#else
#error "other architectures not supported"
#endif
#define unlikely(expr) debugpred__((expr), 0)
#define likely(expr) debugpred__((expr), 1)
#endif
#else
#define unlikely(expr) expr
#define likely(expr) expr
#endif

#if GCC_VERSION >  GCC_VER(3,0,0)
#define MTL_NEVER_NEEDED_INLINE __attribute__ ((always_inline, flatten, visibility("internal")))
#define MTL_PURE_FUNCTION __attribute__ ((pure))
#define MTL_CONST_FUNCTION __attribute__ ((const))
#endif

typedef double v2sd __attribute__ ( ( vector_size ( 16 ) ) );
class Vec2
{
public:
    inline const Vec2& operator= ( const Vec2& rhs )
    {
        v = rhs.v;
        return *this;
    }
    inline void loadpd ( const double *const p )
    {
        v = _mm_load_pd ( p );
    }
    inline void loadupd ( const double *const p )
    {
        v = _mm_loadu_pd ( p );
    }
    inline void storepd ( double *const p ) const
    {
        _mm_store_pd ( p, v );
    }
    inline Vec2 operator+= ( const Vec2& rhs )
    {
        v += rhs.v;
        return *this;
    }
    inline void zero()
    {
        v = _mm_setzero_pd();
    }
    union
    {
        double x[2];
        v2sd v;
    };
};

inline double fast_exp ( double y ) //4% error in [-100; 100 ], using this in the inner loop gives a speedup of 5%
{
    double d;
    * ( ( int* ) ( &d ) + 0 ) = 0;
    * ( ( int* ) ( &d ) + 1 ) = ( int ) ( 1512775 * y + 1072632447 );
    return d;
}

/** Request a chunk of aligned memory that is aligned to the Cache Line size as given by CLS
 * @param mem A pointer to the memory location.
 * @param size The size in bytes of the rquested memory.
 */
template <typename T>
void getCLSalignedmem( T** mem, size_t size)
{
    int status = posix_memalign ( (void**) mem, CLS, size );
    if ( unlikely ( status != 0 ) ) throw std::runtime_error ( "failed to allocate memory!" );
}

/**
A class to store large amounts of data, while keeping the number of page faults on the second index(nt)
small. Therefore every line of nw is (ideally) on a seperate page
It is not whorthwhile to use it for the xker_stor array. performance degrades seriously(30s->39s)
The number of TLB misses stays the same though...
The type T should be reasonable enough to be a divider of 64 (thereofore sth. like 2,4,16,32,64)
*/
template <class T>
class PAARR_2D_dynamic_small//Page-aligned-array...with two indices
{
public:
    inline PAARR_2D_dynamic_small ( uint nw_max, uint nt_max )
    {
//calculate the memory one whole nw line takes, if we use proper CL alignment.
        size_of_nw_line = ( ( nt_max*sizeof ( T ) %CLS == 0 ) ? nt_max*sizeof ( T ) : ( nt_max*sizeof ( T ) /CLS +1 ) *CLS );
        nw_indices_per_page = page_size/size_of_nw_line;
        uint nr_of_pages = nw_max/nw_indices_per_page;
        std::cout<<"parameters of PAARR_2D_small nr_of_pages = "<<nr_of_pages<<" indices_per_page="<<nw_indices_per_page<<" size_of_line="<<size_of_nw_line<<" pagesize = "<<page_size<<std::endl;
        std::cout<<"Requesting "<<nr_of_pages*page_size/1024.0/1024.0<<" MBs of memory"<<std::endl;
        std::cout<<"sizeof user data: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0<<std::endl;
        std::cout<<"Percentage used: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0/ ( nr_of_pages*page_size/1024.0/1024.0 ) *100.0<<"%"<<std::endl;
        getCLSalignedmem(&mem, nr_of_pages* page_size);
    }
    inline ~PAARR_2D_dynamic_small()
    {
        free ( mem );
    }
    inline T *const operator[] ( uint nw )
    {
//determine page...
        uint page = nw/nw_indices_per_page;//getting rid of this division yields a speedup of 3%...
//offset into that page
        uint offs_into_page = nw % nw_indices_per_page;
        uint page_offset = page<<page_size_exponent;//gcc can optimize this multiplication itself, if page_size is a true literal
//return address using byte based numbers
        return ( T* ) ( ( char* ) mem + page_offset + offs_into_page*size_of_nw_line );
    }
private:
    T* mem;
    uint size_of_nw_line;
    uint nw_indices_per_page;
};

template <class T, uint nw_indices_per_page>
class PAARR_2D_small//Page-aligned-array...with two indices
{
public:
    inline PAARR_2D_small ( uint nw_max, uint nt_max )
    {
//calculate the memory one whole nw line takes, if we use proper CL alignment.
        size_of_nw_line = ( ( nt_max*sizeof ( T ) %CLS == 0 ) ? nt_max*sizeof ( T ) : ( nt_max*sizeof ( T ) /CLS +1 ) *CLS );
//        nw_indices_per_page = page_size/size_of_nw_line;
        uint nr_of_pages = nw_max/nw_indices_per_page;
        std::cout<<"parameters of static PAARR_2D_small nr_of_pages = "<<nr_of_pages<<" indices_per_page="<<nw_indices_per_page<<" pagesize = "<<page_size<<std::endl;
        std::cout<<"Requesting "<<nr_of_pages*page_size/1024.0/1024.0<<" MBs of memory"<<std::endl;
        std::cout<<"sizeof user data: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0<<std::endl;
        std::cout<<"Percentage used: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0/ ( nr_of_pages*page_size/1024.0/1024.0 ) *100.0<<"%"<<std::endl;
        getCLSalignedmem(&mem, nr_of_pages* page_size);
    }
    inline ~PAARR_2D_small()
    {
        free ( mem );
    }
    inline T *const operator[] ( uint nw )
    {
//determine page...
        uint page = nw/nw_indices_per_page;//getting rid of this division yields a speedup of 3%...
//offset into that page
        uint offs_into_page = nw % nw_indices_per_page;
        uint page_offset = page<<page_size_exponent;//gcc can optimize this multiplication itself, if page_size is a true literal
//return address using byte based numbers
        return ( T* ) ( ( char* ) mem + page_offset + offs_into_page*size_of_nw_line );
    }
private:
    T* mem;
    uint size_of_nw_line;
//    uint nw_indices_per_page;
};

template <class T>
class PAARR_2D_dynamic_large//Page-aligned-array...with two indices
{
public:
    PAARR_2D_dynamic_large ( uint nw_max, uint nt_max )
    {
      pages_per_index = (nt_max*sizeof(T)/page_size + 1);
        uint size_of_nw_line = pages_per_index*page_size;
        uint nr_of_pages = nw_max*pages_per_index;
        std::cout<<"parameters of PAARR_2D_large nr_of_pages = "<<nr_of_pages<<" pages per index: "<<pages_per_index<<" size_of_line="<<size_of_nw_line<<" pagesize = "<<page_size<<std::endl;
        std::cout<<"Requesting "<<nr_of_pages*page_size/1024.0/1024.0<<" MBs of memory"<<std::endl;
        std::cout<<"sizeof user data: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0<<std::endl;
        std::cout<<"Percentage used: "<<nw_max*nt_max*sizeof ( T ) /1024.0/1024.0/ ( nr_of_pages*page_size/1024.0/1024.0 ) *100.0<<"%"<<std::endl;
        getCLSalignedmem(&mem, nr_of_pages* page_size);
    }
    ~PAARR_2D_dynamic_large()
    {
        free ( mem );
    }
    T *const operator[] ( uint nw )
    {
//determine page...
        uint page = nw*pages_per_index;
//offset of page
        uint page_offset = page<<page_size_exponent;//gcc can optimize this multiplication itself, if page_size is a true literal
//return address using byte based numbers
        return ( T* ) ( ( char* ) mem + page_offset);
    }
private:
    T* mem;
    uint pages_per_index;
};

#ifdef __GNUC__
static inline double ranf ( uint& seed ) __attribute__ ( ( nothrow, pure ) );
static inline int ri ( int high, uint& seed ) __attribute__ ( ( nothrow, pure ) );
static inline double PhiM1 ( double x, const double& om_st, const double& om_end ) __attribute__ ( ( nothrow, pure ) );
static inline int NPhiM1 ( double x, double om_st_1, double om_en_1, double dom ) __attribute__ ( ( nothrow, const ) );
static inline double xpbc ( double x, double xl ) __attribute__ ( ( nothrow, const ) );
static inline void sum_xn_boxes ( double *const __restrict__ xn_m, const Vec2 *const __restrict__ xn, uint ndis, uint ngamma )  __attribute__ ( ( nonnull ( 1, 2 ) ) );
template<bool CLS_OPTIMAL, class PAARR> 
static double cmc ( const uint ntau, double *const __restrict__ xqmc1, const double *const __restrict__ xtau,
                     PAARR& xker_table, Vec2 *const __restrict__ xn, const double alpha,
                    const uint nsweeps, double *const __restrict__ xn_m, double& En_m, double& Acc_1, double& Acc_2,
                    const double deltaxmax, const double delta2, const double om_st_1, const double om_en_1,
                    const double invdom, uint& iseed, const uint omega_points, const uint ngamma, double *const __restrict__ xker_stor, int opti_ntau,
                    double *const __restrict__ h ) __attribute__ ( ( nonnull ( 2, 3, 5, 8, 20, 22 ) ) );
#endif

double normalization = 1.0/static_cast<double> ( ( std::numeric_limits< uint >::max()-1 ) );
static inline double ranf ( uint& seed ) //Note that on my system the uint is 4 bytes in size... if its different on yours other choices for the RNG should be better
{
    /*
           minstd_rand lcg ( seed );
           std::uniform_real_distribution<double> dis;
           double retval = dis ( lcg );
           stringstream temp;
           temp << lcg;
           temp >> seed;
           return retval;*/
    /* A try at Imadas RNG...
    seed = seed * 48828125;
    if (seed < 0)
    seed = (seed + 2147483647) + 1;
    return static_cast<double>(seed%2147483647u) / 2147483647.0;*/
    seed = ( 62089911 * seed + 4349 ) % ( std::numeric_limits< uint >::max()-1 ); //A good LCG, according to Knuth(I think)... and well optimized by gcc
    return static_cast<double> ( seed ) /*/static_cast<double> ( ( std::numeric_limits< uint >::max()-1 ) )*/ * normalization;
}

static inline Vec2 ranf2 ( uint& seed ) //Note that on my system the uint is 4 bytes in size... if its different on yours other choices for the RNG should be better
{
    //optimization for generating two random floats. saves a division.
    //But it didn't work out gcc generates too much reshuffling to get the result to the place it wants them
    Vec2 retval;
    seed = ( 62089911 * seed + 4349 ) % ( std::numeric_limits< uint >::max()-1 ); //A good LCG, according to Knuth(I think)... and well optimized by gcc
    retval.x[0] = static_cast<double> ( seed );
    seed = ( 62089911 * seed + 4349 ) % ( std::numeric_limits< uint >::max()-1 ); //A good LCG, according to Knuth(I think)... and well optimized by gcc
    retval.x[1] = static_cast<double> ( seed );
    retval.v /= _mm_set1_pd ( static_cast<double> ( ( std::numeric_limits< uint >::max()-1 ) ) );
    return  retval;
}

static inline int ri ( int high, uint& seed )
{
    /*
       minstd_rand lcg ( seed );
       std::uniform_int_distribution<> dis ( 0, high );
       int retval = dis ( lcg );
       stringstream temp;
       temp << lcg;
       temp >> seed;*/
    return std::lround ( ranf ( seed ) *high );
}

static void condsqrt ( double& a )
{
    if ( a > 0 )
        a = sqrt ( a );
    else
        a = 0;
}

static inline double PhiM1 ( double x, const double& om_st, const double& om_end )
{
    return x* ( om_end - om_st ) + om_st;
}

static inline int NPhiM1 ( double x, double om_st_1, double om_en_1, double invdom )
{
//flat default with sum 1. This is the correct sum rule for the data
// D(om) = 1.0/(om_en_1 - om_st_1)
    double om = x * ( om_en_1 - om_st_1 ) + om_st_1;
    return std::lround ( ( om-om_st_1 ) * invdom - 0.25 );//subtracting 0.25 rids us of one integer addition. It would be more worthwhile to get rid of the subtraction by 0.25
    //This function yields C compatible indices
    /** sifting through the gfortran source, one finds that NINT is implemented as rounding with rounding mode RND_ROUND, instead of TRUNC, CEIL, or FLOOR.
     * Therefore I assume that lround most closely matches its semantics
     * std::lround ( ( om-om_st_1 ) * invdom - 0.25 ) <=> std::lround ( ( om-om_st_1 ) * invdom + 0.75 ) - 1
     * */
}

static inline double xpbc ( double x, double xl )
{
    double retval = x;
    if ( x > xl ) retval = x - xl;
    if ( x < 0 ) retval = x + xl;
    return retval;
}

static inline void sum_xn_boxes ( double *const __restrict__ xn_m, const Vec2 *const __restrict__ xn, uint ndis, uint ngamma )
{
    for ( uint ng = 0; ng < ngamma; ++ng )
    {
        //gcc is unable to vectorize this loop
        Vec2 t = xn[ng];
        int nd = static_cast<int> ( std::trunc ( ndis * t.x[0] ) ); //saves an integer subtraction and a FP addition.
        //we seem to have the equivalence std::lround ( ndis * t.x[0] + 0.5 )-1 == int(std::trunc ( ndis * t.x[0]))
        xn_m[nd] += t.x[1];
    }
}

template <bool CLS_OPTIMAL>
struct CLS_Trait
{
    inline CLS_Trait ( uint ntau ) : ntau ( ntau ), maxclstau ( ntau ) {}
    inline void setup_x_loop ( const double*, double*, double*, double ) const
    {
    }
    inline void setup_h_loop ( double*, double*, double* ) const
    {
    }
    inline void lambda0_remainder ( const double* const, const double* const, const double *const, double* const, const double, double& ) const {}
    inline void lambda1_remainder ( const double *const, const double *const, const double* const, double*, double, double, double& ) const
    {}
    inline void move_accepted_loop ( const double*, double* ) const {}
    inline void en_loop ( double&, const double *const ) const {}
    uint ntau;
    uint maxclstau;
};

template <>
struct CLS_Trait<false>
{
    inline CLS_Trait ( uint ntau ) : ntau ( ntau )
    {
        maxclstau = ( ntau/ ( CLS/sizeof ( double ) ) ) * ( CLS/sizeof ( double ) );
    }
    inline void setup_x_loop ( const double* xkt_ptr_start, double* psks_start, double* deltah, double zgamma ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
        {
            double temp = xkt_ptr_start[nt];
            psks_start[nt] = temp;
            deltah[nt] += temp * zgamma;//we abuse deltah here to store the intermediate x values
        }
    }
    inline void setup_h_loop ( double* h, double* deltah, double* xqmc1 ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
        {
            h[nt] = deltah[nt] - xqmc1[nt];
        }
    }
    inline void lambda1_remainder ( const double *const xker_stor_ng1, const double *const xker_stor_ng2, const double* const h, double* deltah, double diff0, double diff1, double& deltaE ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
        {
            double temp = xker_stor_ng1[nt]*diff0 + xker_stor_ng2[nt] * diff1;
            deltaE += ( 2.0*h[nt] + temp ) *temp;
            deltah[nt] = temp;
        }
    }
    inline void lambda0_remainder ( const double* const pxkn_start, const double* const pxks_start, const double *const h, double* const deltah, const double z_gamma_o0, double& deltaE ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
        {
            double temp = ( /*xker_table[nw*opti_ntau + nt]*/* ( pxkn_start + nt ) - pxks_start[nt] ) * z_gamma_o0; //that first pointer arithmetic gives better code by gcc
            deltaE += ( 2.0*h[nt] + temp ) *temp;
            deltah[nt] = temp;
        }
    }
    inline void move_accepted_loop ( const double* deltah, double* h ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
            h[nt] += deltah[nt];
    }
    inline void en_loop ( double& En, const double *const h ) const
    {
        for ( uint nt = maxclstau; nt < ntau; ++nt )
            En += h[nt]* h[nt];
    }
    uint ntau;
    uint maxclstau;
};

template<bool CLS_OPTIMAL>
static inline void move_accepted ( const int lambda_max, int& NAcc_1, int& NAcc_2, double* xker_stor_ng1, const double *const xker_table_ptr, uint lambda[2], Vec2 *const xn,
                                   double a_gamma_p[2], double z_gamma_p[2], const double *const deltah, double *const h, const CLS_Trait<CLS_OPTIMAL>& cls_trait )
{
    if ( lambda_max == 0 )
    {
        ++NAcc_1;
        memcpy ( xker_stor_ng1, xker_table_ptr, cls_trait.ntau * sizeof ( double ) );//we only copy over that data if the move was chosen and accepted
    }
    else
        ++NAcc_2;
    for ( int nl = 0; nl <= lambda_max; ++nl )
    {
        xn[lambda[nl]].x[0] = a_gamma_p[nl];
        xn[lambda[nl]].x[1] = z_gamma_p[nl];
    }
    const double* pdh = deltah;//necessary to get gcc to generate good code
    double* ph = h;
    for ( uint nt = 0; nt < cls_trait.maxclstau; nt += 8, pdh += 8, ph += 8 )
    {
        Vec2 dh[4];
        Vec2 hh[4];
        for ( uint t = 0; t < 4; ++t )
        {
            dh[t].loadpd ( pdh + 2*t );
            hh[t].loadpd ( ph + 2*t );
            hh[t] += dh[t];
            hh[t].storepd ( ph + 2*t );
        }
    }
    cls_trait.move_accepted_loop ( deltah, h );
}

template <bool CLS_OPTIMAL, class PAARR>
static double cmc ( const uint ntau, double *const __restrict__ xqmc1, const double *const __restrict__ xtau, PAARR& xker_table, Vec2 *const __restrict__ xn, const double alpha,
                    const uint nsweeps, double *const __restrict__ xn_m, double& En_m, double& Acc_1, double& Acc_2,
                    const double deltaxmax, const double delta2, const double om_st_1, const double om_en_1,
                    const double invdom, uint& iseed, const uint omega_points, const uint ngamma, double *const __restrict__ xker_stor, int opti_ntau,
                    double *const __restrict__ h )
{
    CLS_Trait<CLS_OPTIMAL> cls_trait ( ntau );
//setup h(tau)
    double* pxqmc1 = xqmc1;
    double * __restrict__ deltah;
    posix_memalign ( ( void** ) ( &deltah ), CLS, ntau*sizeof ( double ) );
#if GCC_VERSION >= GCC_VER(4,7,0)
//    h = (double *const)__builtin_assume_aligned(h, 64);
    deltah = (double*)__builtin_assume_aligned(deltah, 64);
//    xker_stor = (double *const)__builtin_assume_aligned(xker_stor, 64);
#endif
    double* phhh = h;
    memset ( deltah, 0, sizeof ( double ) * ntau );
    for ( uint ng = 0; ng < ngamma; ++ng ) //xn is 16byte aligned, xker_stor is 64 byte aligned, h is 64 byte laigned, xker_table is 64byte aligned
    {
        Vec2 az = xn[ng];
        int idx = NPhiM1 ( az.x[0], om_st_1, om_en_1, invdom );
        const double* xkt_ptr_start =  xker_table[idx];
        const double* xkt_ptr = xkt_ptr_start;
        double* psks_start = & ( xker_stor[ng*opti_ntau] );
        double* psks = psks_start;
        double* dhptr = deltah;
        for ( uint nt = 0; nt < cls_trait.maxclstau; nt += 8, xkt_ptr += 8, psks += 8, dhptr += 8 )
        {
            for ( uint t = 0; t < 4; ++t )
            {
                Vec2 xkt;
                xkt.loadpd ( xkt_ptr + 2*t );
                xkt.storepd ( psks + 2*t );
                xkt.v *= _mm_set1_pd ( az.x[1] );
                xkt.v += _mm_load_pd ( dhptr + 2*t ); //prevents gcc from generating an additional load
                xkt.storepd ( dhptr + 2*t );
            }
        }
        cls_trait.setup_x_loop ( xkt_ptr_start, psks_start, deltah, az.x[1] );
    }
    for ( uint nt = 0; nt < cls_trait.maxclstau; nt += 8 )
    {
        double* ddh = deltah + nt;
        double* pxqmc1 = xqmc1 + nt;
        double* ph = h + nt;
        for ( uint t = 0; t < 4; ++t )
        {
            Vec2 temp;
            temp.loadpd ( ddh + 2*t );
            temp.v -= _mm_loadu_pd ( pxqmc1 + 2*t );
            temp.storepd ( ph + 2*t );
        }
    }
    cls_trait.setup_h_loop ( h, deltah, xqmc1 );

    int NAcc_1 = 0;
    int NAcc_2 = 0;
//    deltah = __builtin_assume_aligned(deltah, CLS);//Not present in gcc 4.6.3
    double En;
    for ( uint nsw = 0; nsw < nsweeps; ++nsw )
    {
        double current_x = ranf ( iseed );
        int current_ng1 = ri ( ngamma-1, iseed );
        double current_x2 = ranf ( iseed );
        for ( uint ng = 0; ng < ngamma; ++ng )
        {
            uint lambda[2] ;
            const int ng1 = current_ng1;
            double x2 = current_x2;
            int next_ng1 = ri ( ngamma-1, iseed );
//	    Vec2 next_xn_ng1 = xn[next_ng1];
            lambda[0] = ng1;
            double next_x = ranf ( iseed );
            double next_x2 = ranf ( iseed );
            int lambda_max;
            Vec2 xn_ng1 = xn[ng1];
            double a_gamma_o[2];
            double z_gamma_o[2] __attribute__ ( ( aligned ( 16 ) ) );
            double a_gamma_p[2];
            double z_gamma_p[2] __attribute__ ( ( aligned ( 16 ) ) );
            int newxker_stor_idx;
            double* pdh = deltah;
            Vec2 ddE[4];
            double deltaE = 0;
            for ( uint t = 0; t < 4; ++t )
                ddE[t].zero();
            if ( current_x > 0.5 )
            {
                //weight sharing moves
                lambda_max = 1;
                int ng2;
                do
                {
                    ng2 = ri ( ngamma-1, iseed );
                }
                while ( ng2 == ng1 );
                lambda[1] = ng2;
                a_gamma_o[0] = xn_ng1.x[0];
                a_gamma_o[1] = xn[ng2].x[0];
                z_gamma_o[0] = xn_ng1.x[1];
                z_gamma_o[1] = xn[ng2].x[1];

                a_gamma_p[0] = xn[ng1].x[0];
                a_gamma_p[1] = xn[ng2].x[0];

                double s = ( z_gamma_o[0] + z_gamma_o[1] ) * x2 - z_gamma_o[0];
                z_gamma_p[0] = z_gamma_o[0] + s;
                z_gamma_p[1] = z_gamma_o[1] - s;
                double diff[2] __attribute__ ( ( aligned ( 16 ) ) );
                diff[0] = z_gamma_p[0] - z_gamma_o[0];
                diff[1] = z_gamma_p[1] - z_gamma_o[1];
                Vec2 temp0;
                Vec2 temp1;
                temp0.x[0] = diff[0];
                temp0.x[1] = diff[0];
                temp1.x[0] = diff[1];
                temp1.x[1] = diff[1];
                double* ph = h;
                for ( int nt = 0; nt < cls_trait.maxclstau; nt += 8, pdh += 8, ph+= 8 )
                {
                    //load cachelines
                    Vec2 xkng1[4];
                    Vec2 xkng2[4];
                    for ( int t = 0; t < 4; ++t )
                    {
                        xkng1[t].loadpd ( &xker_stor[ng1*opti_ntau] + nt + 2*t );
                        xkng2[t].loadpd ( &xker_stor[ng2*opti_ntau] + nt + 2*t );
                        xkng1[t].v *= temp0.v;
                        xkng2[t].v *= temp1.v;
                        xkng1[t].v += xkng2[t].v;//xkng1[t] stores the content of deltah
                        Vec2 temp;
                        temp.loadpd ( ph + 2*t );
                        temp.v += temp.v;
                        temp.v += xkng1[t].v;
                        temp.v *= xkng1[t].v;
                        ddE[t].v += temp.v;
                    }
                    xkng1[0].storepd ( pdh + 0 );
                    xkng1[1].storepd ( pdh + 2 );
                    xkng1[2].storepd ( pdh + 4 );
                    xkng1[3].storepd ( pdh + 6 );
                }
                cls_trait.lambda1_remainder ( xker_stor + ng1*opti_ntau, xker_stor + ng2*opti_ntau, h, deltah, diff[0], diff[1], deltaE );
            }
            else
            {
                lambda_max = 0;
                z_gamma_o[0] = xn_ng1.x[1];
                z_gamma_p[0] = xn_ng1.x[1];
                a_gamma_o[0] = xn_ng1.x[0];
                a_gamma_p[0] = xpbc ( xn_ng1.x[0] + ( x2 - 0.5 ) *deltaxmax, 1.0 );

                uint nw = NPhiM1 ( a_gamma_p[0], om_st_1, om_en_1, invdom );//EVIL!
                const double* pxkn_start = xker_table[nw];
                const double* pxkn = pxkn_start;
                newxker_stor_idx = nw;
                Vec2 xkng1[4];
                Vec2 xkn[4];
                Vec2 zgo;
                zgo.x[0] = z_gamma_o[0];
                zgo.x[1] = z_gamma_o[0];
                const double* pxks_start = & ( xker_stor[ng1*opti_ntau] );
                const double* pxks = pxks_start;
                const double* ph = h;
                for ( uint nt = 0; nt < cls_trait.maxclstau; nt += 8, pdh += 8, pxks += 8, pxkn += 8, ph += 8 )
                {
                    for ( int t = 0; t < 4; ++t )
                    {
                        xkng1[t].loadpd ( pxks + 2*t );//pxks gives cache misses
                        xkn[t].loadpd ( pxkn + 2*t );
                        xkn[t].v -= xkng1[t].v;
                        xkn[t].v *= zgo.v;//xkn[t] stores the content of deltah
                        Vec2 temp;
                        temp.loadpd ( ph + 2*t );
                        temp.v += temp.v;
                        temp.v += xkn[t].v;
                        temp.v *= xkn[t].v;
                        ddE[t].v += temp.v;
                        xkn[t].storepd ( pdh + 2*t );
                    }
                }
                cls_trait.lambda0_remainder ( pxkn_start, pxks_start, h, deltah, z_gamma_o[0], deltaE );
//		            asm("nop;nop;nop;");
//                 for ( uint nt = cls_trait.maxclstau; nt < ntau; ++nt )
//                 {
//                     double temp = ( /*xker_table[nw*opti_ntau + nt]*/* ( pxkn_start + nt ) - pxks_start[nt] ) * z_gamma_o[0]; //that first pointer arithmetic gives better code by gcc
//                     deltaE += ( 2.0*h[nt] + temp ) *temp;
//                     deltah[nt] = temp;
//                 }
            }
            ddE[0].v += ddE[1].v;
            ddE[2].v += ddE[3].v;
            ddE[0].v += ddE[2].v;
//            ddE[0].v = _mm_hadd_pd ( ddE[0].v, ddE[0].v );
//	    deltaE += ddE[0].x[0];
            deltaE = deltaE + ddE[0].x[0] + ddE[0].x[1];
            if ( deltaE < 0 ) //move definitely accepted
                move_accepted<CLS_OPTIMAL> ( lambda_max, NAcc_1, NAcc_2, & xker_stor[ng1*opti_ntau], xker_table[newxker_stor_idx], lambda, xn,
                                             a_gamma_p, z_gamma_p, deltah, h, cls_trait );
            else
            {
                //check harder
                double ratio_x = ranf ( iseed );
                double ratio = exp ( -alpha * deltaE );
                if ( ratio > /*ranf ( iseed )*/ratio_x ) //move accepted
                    move_accepted<CLS_OPTIMAL> ( lambda_max, NAcc_1, NAcc_2, & xker_stor[ng1*opti_ntau], xker_table[newxker_stor_idx], lambda, xn,
                                                 a_gamma_p, z_gamma_p, deltah, h, cls_trait );
            }
            current_x = next_x;
            current_ng1 = next_ng1;
            current_x2 = next_x2;
//	    current_xn_ng1 = next_xn_ng1;
        }//end ng loop
        En = 0;
        Vec2 dEn[4];
        for ( int t = 0; t < 4; ++t )
            dEn[t].zero();
        double* ph = h;//the pointer for gcc...
        for ( uint nt = 0; nt < cls_trait.maxclstau; nt += 8, ph += 8 )
        {
            for ( uint t = 0; t < 4; ++t )
            {
                Vec2 temp;
                temp.loadpd ( ph + 2*t );
                temp.v *= temp.v;
                dEn[t] += temp;
            }
        }
        dEn[0] += dEn[1];
        dEn[2] += dEn[3];
        dEn[0] += dEn[2];
        En = dEn[0].x[0] + dEn[0].x[1];
        cls_trait.en_loop ( En, h );
        En_m += En;
        sum_xn_boxes ( xn_m, xn, omega_points, ngamma );
    }
    Acc_1 = static_cast<double> ( NAcc_1 ) /static_cast<double> ( ngamma*nsweeps );
    Acc_2 = static_cast<double> ( NAcc_2 ) /static_cast<double> ( ngamma*nsweeps );
    En_m /= static_cast<double> ( nsweeps );
    for ( uint nd = 0; nd < omega_points; ++nd )
        xn_m[nd] /= static_cast<double> ( nsweeps );
    free ( deltah );
    return En;
}

template <class T>
struct Container_2D
{
  inline Container_2D(uint d1, uint d2) : mem(new T[d1*d2]), dim1(d1), dim2(d2)
  {}
  inline ~Container_2D()
  {
    delete [] mem;
  }
  inline const T& operator()(uint i, uint k) const
  {//no checking... just nice semantics
    return mem[i * dim2 + k];
  }
  inline T& operator()(uint i, uint k)
  {//no checking... just nice semantics
    return mem[i * dim2 + k];
  }
  inline T* operator[](uint i)
  {
    return mem + i * dim2;
  }
  inline void zero()
  {
    memset ( mem, 0, dim1*dim2*sizeof (T) );
  }
  T *const mem;
  const int dim1;
  const int dim2;
};

template <bool CLS_OPTIMAL, class PAARR>
int cmaxent_pt ( double *const __restrict__ xqmc, const double *const xtau, const int len,
                       double xmom1, double ( *xker ) ( const double&, double&, double& ), double ( *backtrans ) ( double&, double&, double& ),
                       double beta, double *const alpha_tot, const int n_alpha, const uint ngamma, const double omega_start, const double omega_end,
                       const uint omega_points, int nsweeps, int nbins, uint nwarmup, const std::string dump_Aom, std::string max_stoch_log,
		       std::string dump, const double *const __restrict__ u, double *const __restrict__ sigma, Container_2D<Vec2>& xn_tot,
		       double *const __restrict__ xn_m, double *const __restrict__ xn_e, double *const __restrict__ En_m_tot, double *const __restrict__ En_e_tot,
		       Container_2D<double>& xn_m_tot, Container_2D<double>& xn_e_tot )
{
    const double DeltaXMAX = 0.01;
    const double delta = 0.001;
    const double delta2 = delta*delta;
    const uint ntau = len;
    const uint nsims = n_alpha;
    double om_st_1 = omega_start;
    double om_en_1 = omega_end;
    const int ndis_table = 50000;
    const double invdom = static_cast<double> ( ndis_table -1 ) / ( om_en_1 - om_st_1 ); //not sure about that -1
    const uint opti_ntau = ( ntau% ( CLS/sizeof ( double ) ) == 0 ) ? ntau : ( ntau / ( CLS/sizeof ( double ) ) +1 ) * ( CLS/sizeof ( double ) );
    PAARR xker_table ( ndis_table, ntau );
    for ( uint nw = 0; nw < ndis_table; ++nw )
    {
        double* xker_table_ptr = xker_table[nw];
        for ( uint nt = 0; nt < ntau; ++nt )
        {
            double om = omega_start + nw/invdom;//better for cmc
            /*xker_table[nw*opti_ntau + nt]*/ xker_table_ptr[nt] = xker ( xtau[nt], om, beta );
        }
    }
//normalize data
    for ( int i = 0; i < len; ++i )
    {
        xqmc[i] /= xmom1;
//we take care of the normalization of the covariance matrix later, in the eigenvalues
    }

    for ( int nt = 0; nt < len; ++nt )
    {
        //If I divide the full cov matrix by xmom^2 then the eigenvalues change by that. The eigenvectors are unchanged
        //and therefore the expression below by |xmom|
        sigma[nt] = sqrt ( sigma[nt] ) /std::fabs ( xmom1 );
    }
    double xqmc1[len];
    if ( u != NULL )
    {
        memset ( xqmc1, 0, len*sizeof ( double ) );
        for ( uint nt1 = 0; nt1 < ntau; ++nt1 )
        {
            for ( uint nt = 0; nt < ntau; ++nt )
                xqmc1[nt1] += xqmc[nt]* u[nt1 * len + nt];
            xqmc1[nt1] /= sigma[nt1];
        }
        double vhelp[len];
        for ( uint nw = 0; nw < ndis_table; ++nw )
        {
            memset ( vhelp, 0, len* sizeof ( double ) );
            double* xker_table_ptr = xker_table[nw];
            for ( int nt1 = 0; nt1 < len; ++nt1 )
                for ( int nt = 0; nt < len; ++nt )
                {
                    vhelp[nt1] += xker_table_ptr[nt]*u[nt1 * len + nt];
                }

            for ( int nt1 = 0; nt1 < len; ++nt1 )
                xker_table_ptr[nt1] = vhelp[nt1]/sigma[nt1];
        }
    }//end branch for a present cov
    else
    {
        //diagonal cov. errors are in sigma
        for ( uint nt1 = 0; nt1 < ntau; ++nt1 )
            xqmc1[nt1] = xqmc[nt1]/sigma[nt1];
        for ( uint nw = 0; nw < ndis_table; ++nw )
        {
            double* xker_table_ptr = xker_table[nw];
            for ( uint nt1 = 0; nt1 < len; ++nt1 )
                xker_table_ptr[nt1] /= sigma[nt1];
        }
    }
    double D = 1.0 / ( om_en_1 - om_st_1 );
//now create files...
    std::fstream confdump ( dump.c_str(), ios::in ); //fd 41 in Fakher's Maxent
    std::fstream Aomdump ( dump_Aom.c_str(), ios::in ); //fd 42 in Fakher's Maxent
    int nc = 0;
    uint iseed = 8752143;
    if ( confdump && Aomdump )
    {
        //previous data present
        nwarmup = 0;
        confdump>>iseed;
        for ( uint ns = 0; ns < nsims; ++ns )
        {
            for ( uint ng = 0; ng < ngamma; ++ng )
            {
                confdump>>xn_tot[ns][ng].x[0];
                confdump>>xn_tot[ns][ng].x[1];
            }
            confdump>>En_m_tot[ns];
            confdump>>En_e_tot[ns];
        }
        Aomdump>>nc;
        for ( uint ns = 0; ns < nsims; ++ns )
            for ( uint nd = 0; nd < omega_points; ++nd )
            {
                confdump>>xn_m_tot[ns][nd];
                confdump>>xn_e_tot[ns][nd];
            }
        std::ofstream stochlog ( max_stoch_log.c_str() );
        stochlog<<"Read from dump: nc = "<<nc<<std::endl;
    }
    else
    {
        //no previous data present
        for ( uint ns = 0; ns < nsims; ++ns )
            for ( uint ng = 0; ng < ngamma; ++ng )
            {
                xn_tot[ns][ng].x[0] = ranf ( iseed );
                xn_tot[ns][ng].x[1] = 1.0/static_cast<double> ( ngamma );
            }
        xn_m_tot.zero();
	xn_e_tot.zero();
//        memset ( xn_m_tot, 0, nsims*ngamma*sizeof ( double ) );
        memset ( En_m_tot, 0, nsims*sizeof ( double ) );
//        memset ( xn_e_tot, 0, nsims*ngamma*sizeof ( double ) );
        memset ( En_e_tot, 0, nsims*sizeof ( double ) );
        std::ofstream stochlog ( max_stoch_log.c_str() );
        stochlog<<"No dump data"<<std::endl;
    }
    confdump.close();
    Aomdump.close();
    //start simulations
    double En_tot[nsims];
    double* xker_stor;
    double* h;
    posix_memalign ( ( void** ) ( &xker_stor ), CLS, ngamma*opti_ntau*sizeof ( double ) );
    posix_memalign ( ( void** ) ( &h ), CLS, ntau*sizeof ( double ) );
#if GCC_VERSION >= GCC_VER(4,7,0)
    h = (double*)__builtin_assume_aligned(h, 64);
    xker_stor = (double*)__builtin_assume_aligned(xker_stor, 64);
#endif
    for ( uint nb = 0; nb < nbins; ++nb )
    {
        std::ofstream stochlog ( max_stoch_log.c_str(), ios::app );//move opening of filestream here
        stochlog.precision ( 7 );
        for ( int ns = 0; ns < nsims; ++ns )
        {
            double alpha = alpha_tot[ns];
            double Acc_1, Acc_2;
            double En_m = 0;
            memset ( xn_m, 0, sizeof ( double ) *omega_points );
//            En_tot[ns] = cmc<CLS_OPTIMAL> ( ntau, xqmc1, xtau, xker_table, xn_tot[ns], alpha, nsweeps, xn_m, En_m, Acc_1, Acc_2,
//                                            DeltaXMAX, delta2, om_st_1, om_en_1, invdom, iseed, omega_points, ngamma, xker_stor , opti_ntau, h );
            //the result is the energy of the configuration xn_tot for simulation ns
            stochlog<<"Alpha, En_m, Acc "<< 1.0/alpha<<" "<<En_m<<" "<<Acc_1<<" "<<Acc_2<<std::endl;//do not use scientific here
            if ( likely ( nb > nwarmup ) )
            {
//	      std::cout<<En_tot[ns]<<" "<<En_m<<std::endl;
                if ( ns == 0 ) ++nc;
                for ( uint nd = 0; nd < omega_points; ++nd )
                {
                    xn_m[nd] *= ( D * static_cast<double> ( omega_points ) );
                    xn_m_tot[ns][nd] += xn_m[nd];
                    xn_e_tot[ns][nd] += xn_m[nd] * xn_m[nd];
                }
                En_m_tot[ns] += En_m;
                En_e_tot[ns] += En_m * En_m;
            }
        }//end ns loop
//Exchange
        double Acc_1 = 0;
        for ( uint nex = 0; nex < 2*nsims; ++nex )
        {
            const uint nalp1 = ri ( nsims - 2, iseed ); //[0, nsims-2]
            const uint nalp2 = nalp1 + 1;
            double deltaE = ( alpha_tot[nalp1] * En_tot[nalp2] + alpha_tot[nalp2]* En_tot[nalp1] )
                            - ( alpha_tot[nalp1] * En_tot[nalp1] + alpha_tot[nalp2]* En_tot[nalp2] );
            double ratio = std::exp ( -deltaE );
            if ( ratio > ranf ( iseed ) )
            {
                Acc_1 += 1.0;
                swap_ranges ( xn_tot[nalp1], xn_tot[nalp1] + ngamma, xn_tot[nalp2] ); //shortest, but gcc can't analyze it
                swap ( En_tot[nalp1], En_tot[nalp2] );
            }
        }//end nex loop
        Acc_1 /= static_cast<double> ( 2*nsims ); //at the end of Fakher's Fortran loop over nex (line 300) nex should have the value 2nsims
        stochlog<<"Acc exchange "<<Acc_1<<std::endl;/*it's just not looking nice in scientific notation...*/
    }//bin loop finished
    free ( xker_stor );
    free ( h );
    std::ofstream stochlog ( max_stoch_log.c_str(), ios::app );
    stochlog.precision ( 7 );
    stochlog<<"Total time: N/A"<<std::endl;
    //dump for restarting purposes
    confdump.open ( dump.c_str(), ios::out ); //fd 41 in Fakher's Maxent
    confdump.precision ( 17 );
    Aomdump.open ( dump_Aom.c_str(), ios::out ); //fd 42 in Fakher's Maxent
    Aomdump.precision ( 17 );
    confdump<<iseed<<std::endl;
    for ( uint ns = 0; ns < nsims; ++ns )
    {
        for ( uint ng = 0; ng < ngamma; ++ng )
        {
            confdump<<std::scientific<<xn_tot[ns][ng].x[0]<<" "<<xn_tot[ns][ng].x[1]<<std::endl;
        }
        confdump<<std::scientific<<En_m_tot[ns]<<" "<<En_e_tot[ns]<<std::endl;
    }
    Aomdump<<nc<<std::endl;
    for ( uint ns = 0; ns < nsims; ++ns )
        for ( uint nd = 0; nd < omega_points; ++nd )
        {
            Aomdump<<std::scientific<<xn_m_tot[ns][nd] <<" "<< xn_e_tot[ns][nd]<<std::endl;
        }
    confdump.close();
    Aomdump.close();
//dumping finished
    return nc;
}

template <bool CLS_OPTIMAL>
inline int cmaxent_pagesplit_selector(uint size_of_nw_line, double *const __restrict__ xqmc, const double *const xtau, const int len,
               double xmom1, double ( *xker ) ( const double&, double&, double& ), double ( *backtrans ) ( double&, double&, double& ),
               double beta, double *const alpha_tot, const int n_alpha, const uint ngamma, const double omega_start, const double omega_end,
               const uint omega_points, int nsweeps, int nbins, uint nwarmup,
               const std::string dump_Aom, std::string max_stoch_log, std::string dump,
               const double *const __restrict__ u, double *const __restrict__ sigma, Container_2D<Vec2>& xn_tot,
	       double *const __restrict__ xn_m, double *const __restrict__ xn_e, double *const __restrict__ En_m_tot, double *const __restrict__ En_e_tot, Container_2D<double>& xn_m_tot, Container_2D<double>& xn_e_tot )
{
  int retval;
switch(page_size/size_of_nw_line)
{
case 1:
 retval = cmaxent_pt<CLS_OPTIMAL, PAARR_2D_small<double, 1> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
break;
case 2:
retval = cmaxent_pt<CLS_OPTIMAL, PAARR_2D_small<double, 2> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
break;
case 4:
retval = cmaxent_pt<CLS_OPTIMAL, PAARR_2D_small<double, 4> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
break;
case 8:
retval = cmaxent_pt<CLS_OPTIMAL, PAARR_2D_small<double, 8> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
break;
default:
            retval = cmaxent_pt<CLS_OPTIMAL, PAARR_2D_dynamic_small<double> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                     nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
break;
}
return retval;
}

/**
 * @param xqmc the tau/i-omega resolved monte carlo data
 * @param xtau the tau/i-omega grid
 * @param len the number of grid points
 * @param xmom1 the first moment
 * @param xker the kernel to use
 * @param backtrans the back-transformation
 * @param omega_points the number of points in the omega spectrum
 * @param nsweeps the number of updates between measurements
 * @param nbins
 * @param nwarmup the number of configurations that get discarded from the beginning
 * @param u the matrix of eigenvectors or a NULL-pointer if the covariance matrix shall be assumed diagonal
 * @param sigma eigenvalues of the covariance matrix or errors
 * */

void cmaxent ( double *const __restrict__ xqmc, const double *const xtau, const int len,
               double xmom1, double ( *xker ) ( const double&, double&, double& ), double ( *backtrans ) ( double&, double&, double& ),
               double beta, double *const alpha_tot, const int n_alpha, const uint ngamma, const double omega_start, const double omega_end,
               const uint omega_points, int nsweeps, int nbins, uint nwarmup,
               const std::string file_root, const std::string dump_Aom, std::string max_stoch_log, std::string energies, std::string best_fit, std::string dump,
               const double *const __restrict__ u, double *const __restrict__ sigma )
{
#ifndef __GNUC__
    page_size_exponent = 0;
    {
        unsigned int v = page_size;

        while ( v >>= 1 )
        {
            r++;
        }
    }
#endif
    const uint& nsims(n_alpha);
    double *const En_m_tot = new double[nsims];
    double *const En_e_tot = new double[nsims];
    Container_2D<Vec2> xn_tot(nsims, ngamma);
    Container_2D<double> xn_m_tot(nsims, omega_points);
    Container_2D<double> xn_e_tot(nsims, omega_points);
    double *const xn_m = new double[omega_points];
    double *const xn_e = new double[omega_points];
    int nc;
//determine the optimal implementation to use
    size_t size_of_nw_line = ( ( len*sizeof ( double ) %CLS == 0 ) ? len*sizeof ( double ) : ( len*sizeof ( double ) /CLS + 1 ) *CLS );
    if ( size_of_nw_line <= page_size )
    {
        if ( len% ( CLS/sizeof ( double ) ) == 0 )
        {
                    //no checks necessary! Go cmaxent, go!
        nc = cmaxent_pagesplit_selector<true>(size_of_nw_line, xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
        }
        else
        {
            //handbrake
        nc = cmaxent_pagesplit_selector<false>(size_of_nw_line, xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
        }
    }
    else
    {
        if ( len% ( CLS/sizeof ( double ) ) == 0 )
        {
            //no checks necessary! Go cmaxent, go!
            nc = cmaxent_pt<true, PAARR_2D_dynamic_large<double> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                    nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
        }
        else
        {
            //handbrake
            nc = cmaxent_pt<false, PAARR_2D_dynamic_large<double> > ( xqmc, xtau, len, xmom1, xker, backtrans, beta, alpha_tot, n_alpha, ngamma, omega_start, omega_end, omega_points, nsweeps, nbins,
                                     nwarmup, dump_Aom, max_stoch_log, dump, u, sigma, xn_tot, xn_m, xn_e, En_m_tot, En_e_tot, xn_m_tot, xn_e_tot );
        }
    }
    std::ofstream energ ( energies.c_str() ); //fd 66 in Fakher's MaxEnt
    energ.precision ( 12 );
    energ << setiosflags ( ios::left );
    for ( uint ns = 0; ns < nsims; ++ns )
    {
        En_m_tot[ns] /= static_cast<double> ( nc );
        En_e_tot[ns] /= static_cast<double> ( nc );
        En_e_tot[ns] = ( En_e_tot[ns] - En_m_tot[ns]*En_m_tot[ns] ) /static_cast<double> ( nc );
        condsqrt ( En_e_tot[ns] );
        energ<<setw ( 14 ) <<alpha_tot[ns]<<setw ( 14 ) <<En_m_tot[ns]<<setw ( 14 ) <<En_e_tot[ns]<<std::endl; //scientific notation doesn't look nice
    }
    energ.close();
    for ( uint ns = 0; ns < nsims; ++ns )
    {
        std::string file1 ( file_root + "_" + to_string ( ns ) );
        std::ofstream aomfile ( file1.c_str() );
        aomfile.precision ( 7 );
        for ( uint nd = 0; nd < omega_points; ++nd )
        {
            xn_m_tot[ns][nd] /= static_cast<double> ( nc );
            xn_e_tot[ns][nd] /= static_cast<double> ( nc );
            xn_e_tot[ns][nd] = ( xn_e_tot[ns][nd] - xn_m_tot[ns][nd]*xn_m_tot[ns][nd] ) / static_cast<double> ( nc );
            condsqrt ( xn_e_tot[ns][nd] );
            double om = PhiM1 ( static_cast<double> ( nd ) /static_cast<double> ( omega_points ), omega_start, omega_end);
            double aom = xn_m_tot[ns][nd] * xmom1;
            double err = xn_e_tot[ns][nd] * xmom1;
            aomfile<<std::scientific<<om<<" "<<backtrans ( aom, beta, om ) <<" "<<backtrans ( err, beta, aom ) <<std::endl;
        }
    }
//Now do the averaging
    for ( uint p_star = 0; p_star < nsims - 10; ++p_star )
    {
        memset ( xn_m, 0, sizeof ( double ) * omega_points );
        memset ( xn_e, 0, sizeof ( double ) * omega_points );
        for ( uint ns = p_star; ns < nsims - 2/*1*/; ++ns )
            for ( uint nd = 0; nd < omega_points; ++nd )
            {
                xn_m[nd] += ( En_m_tot[ns] - En_m_tot[ns + 1] ) * xn_m_tot[ns][nd];
                xn_e[nd] += ( En_m_tot[ns] - En_m_tot[ns + 1] ) * xn_e_tot[ns][nd];
            }
        double multiplier = 1.0/ ( En_m_tot[p_star] - En_m_tot[nsims - 1] );
        for ( uint nd = 0; nd < omega_points; ++nd )
        {
            xn_m[nd] *= multiplier;
            xn_e[nd] *= multiplier;
        }
        std::ofstream aomps ( std::string ( file_root + "_ps_" + to_string ( p_star ) ).c_str() );
        aomps.precision ( 7 );
        double xmax = 0;
        for ( uint nd = 0; nd < omega_points; ++nd )
        {
            double om = PhiM1 ( static_cast<double> ( nd ) /static_cast<double> ( omega_points ), omega_start, omega_end );
            double aom = xn_m[nd] * xmom1;
            double err = xn_e[nd] * xmom1;
            xn_m[nd] = backtrans ( aom, beta, om );
            xn_e[nd] = backtrans ( err, beta, om );
            if ( xn_m[nd] > xmax ) xmax = xn_m[nd];
        }

        for ( uint nd = 0; nd < omega_points; ++nd )
        {
            double om = PhiM1 ( static_cast<double> ( nd ) /static_cast<double> ( omega_points ), omega_start, omega_end );
            aomps<<std::scientific<<om<<" "<<xn_m[nd]<<" "<<xn_e[nd]<<" "<<xn_m[nd]/xmax<<" "<<xn_e[nd]/xmax<<std::endl;
        }
    }
    delete [] En_m_tot;
    delete [] En_e_tot;
    delete [] xn_m;
    delete [] xn_e;
    std::ofstream bf ( best_fit.c_str() );
    bf.precision ( 16 );
    for ( uint ng = 0; ng < ngamma; ++ng )
    {
        bf<<std::scientific<<PhiM1 ( xn_tot[nsims - 1][ng].x[0], omega_start, omega_end ) <<" "<<xn_tot[nsims-1][ng].x[1]<<std::endl;
    }
}
