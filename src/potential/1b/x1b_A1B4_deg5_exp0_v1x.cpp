
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iomanip>

#include "x1b_A1B4_deg5_exp0_v1x.h" 
 

////////////////////////////////////////////////////////////////////////////////

namespace {

//----------------------------------------------------------------------------//

struct variable {
    double v_exp0(const double& r0, const double& k,
                 const double * p1, const double * p2 );
                 
    double v_exp(const double& k,
                 const double * p1, const double * p2 );

    double v_coul0(const double& r0, const double& k,
                  const double * p1, const double * p2 );
                  
    double v_coul(const double& k,
                  const double * p1, const double * p2 );
                  
    void grads(const double& gg, double * grd1, double * grd2,
               const double * p1, const double * p2);

    double g[3]; // diff(value, p1 - p2)
};

//----------------------------------------------------------------------------//

void variable::grads(const double& gg, double * grd1, double * grd2, 
                     const double * p1, const double * p2) {
    for (size_t i = 0; i < 3 ; i++) {
        double d = gg*g[i];
        grd1[i] += d;
        grd2[i] -= d;
    }
}

//----------------------------------------------------------------------------//

double variable::v_exp0(const double& r0, const double& k,
                       const double * p1, const double * p2)
{
    g[0] = p1[0] - p2[0];
    g[1] = p1[1] - p2[1];
    g[2] = p1[2] - p2[2];

    const double r = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

    const double exp1 = std::exp(k*(r0 - r));
    const double gg = - k*exp1/r;

    g[0] *= gg;
    g[1] *= gg;
    g[2] *= gg;

    return exp1;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

double variable::v_coul0(const double& r0, const double& k,
                        const double * p1, const double * p2)
{
    g[0] = p1[0] - p2[0];
    g[1] = p1[1] - p2[1];
    g[2] = p1[2] - p2[2];

    const double rsq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
    const double r = std::sqrt(rsq);

    const double exp1 = std::exp(k*(r0 - r));
    const double rinv = 1.0/r;
    const double val = exp1*rinv;

    const double gg = - (k + rinv)*val*rinv;

    g[0] *= gg;
    g[1] *= gg;
    g[2] *= gg;

    return val;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

double variable::v_exp(const double& k,
                       const double * p1, const double * p2)
{
    g[0] = p1[0] - p2[0];
    g[1] = p1[1] - p2[1];
    g[2] = p1[2] - p2[2];

    const double r = std::sqrt(g[0]*g[0] + g[1]*g[1] + g[2]*g[2]);

    const double exp1 = std::exp(k*(- r));
    const double gg = - k*exp1/r;

    g[0] *= gg;
    g[1] *= gg;
    g[2] *= gg;

    return exp1;
}


//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //


double variable::v_coul(const double& k,
                        const double * p1, const double * p2)
{
    g[0] = p1[0] - p2[0];
    g[1] = p1[1] - p2[1];
    g[2] = p1[2] - p2[2];

    const double rsq = g[0]*g[0] + g[1]*g[1] + g[2]*g[2];
    const double r = std::sqrt(rsq);

    const double exp1 = std::exp(k*(-r));
    const double rinv = 1.0/r;
    const double val = exp1*rinv;

    const double gg = - (k + rinv)*val*rinv;

    g[0] *= gg;
    g[1] *= gg;
    g[2] *= gg;

    return val;
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

struct monomer {
    double oh1[3];
    double oh2[3];

    void setup(const double* ohh,
               const double& in_plane_g, const double& out_of_plane_g,
               double x1[3], double x2[3]);

    void grads(const double* g1, const double* g2,
               const double& in_plane_g, const double& out_of_plane_g,
               double* grd) const;
};

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

void monomer::setup(const double* ohh,
                    const double& in_plane_g, const double& out_of_plane_g,
                    double* x1, double* x2)
{
    for (int i = 0; i < 3; ++i) {
        oh1[i] = ohh[i + 3] - ohh[i];
        oh2[i] = ohh[i + 6] - ohh[i];
    }

    const double v[3] = {
        oh1[1]*oh2[2] - oh1[2]*oh2[1],
        oh1[2]*oh2[0] - oh1[0]*oh2[2],
        oh1[0]*oh2[1] - oh1[1]*oh2[0]
    };

    for (int i = 0; i < 3; ++i) {
        const double in_plane = ohh[i] + 0.5*in_plane_g*(oh1[i] + oh2[i]);
        const double out_of_plane = out_of_plane_g*v[i];

        x1[i] = in_plane + out_of_plane;
        x2[i] = in_plane - out_of_plane;
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

void monomer::grads(const double* g1, const double* g2,
                    const double& in_plane_g, const double& out_of_plane_g,
                    double* grd) const
{
    const double gm[3] = {
        g1[0] - g2[0],
        g1[1] - g2[1],
        g1[2] - g2[2]
    };

    const double t1[3] = {
        oh2[1]*gm[2] - oh2[2]*gm[1],
        oh2[2]*gm[0] - oh2[0]*gm[2],
        oh2[0]*gm[1] - oh2[1]*gm[0]
    };

    const double t2[3] = {
        oh1[1]*gm[2] - oh1[2]*gm[1],
        oh1[2]*gm[0] - oh1[0]*gm[2],
        oh1[0]*gm[1] - oh1[1]*gm[0]
    };

    for (int i = 0; i < 3; ++i) {
        const double gsum = g1[i] + g2[i];
        const double in_plane = 0.5*in_plane_g*gsum;

        const double gh1 = in_plane + out_of_plane_g*t1[i];
        const double gh2 = in_plane - out_of_plane_g*t2[i];

        grd[i + 0] += gsum - (gh1 + gh2); // O
        grd[i + 3] += gh1; // H1
        grd[i + 6] += gh2; // H2
    }
}

//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - //

//struct vsites {
//    //void TwoParticleAverageSite() {}
//    //void ThreeParticleAverageSite() {}
//    void OutOfPlaneSite(const double& w12, const double& w13,
//                        const double& wcross, const double x1[3],
//                        const double y1[3], const double y2[3],
//                        double vs[3]);
//    //void LocalCoordinatesSite{}
//};
//
//void vsites::OutOfPlaneSite(const double& w12,
//                            const double& w13,
//                            const double& wcross,
//                            const double x1[3],
//                            const double y1[3],
//                            const double y2[3],
//                            double vs[3]) {
//    double r12[3], r13[3];
//
//    for (int i = 0; i < 3; ++i) {
//        r12[i] = y1[i] - x1[i];
//        r13[i] = y2[i] - x1[i];
//    }
//                            
//    double rc[3];
//    rc[0] = r12[1]*r13[2] - r12[2]*r13[1];
//    rc[1] = r12[2]*r13[0] - r12[0]*r13[2];
//    rc[2] = r12[0]*r13[1] - r12[1]*r13[0];
//    
//    vs[0] = x1[0] + w12 * r12[0] + w13 * r13[0] + wcross * rc[0];
//    vs[1] = x1[1] + w12 * r12[1] + w13 * r13[1] + wcross * rc[1];
//    vs[2] = x1[2] + w12 * r12[2] + w13 * r13[2] + wcross * rc[2];
//}

} // namespace

////////////////////////////////////////////////////////////////////////////////

namespace x1b_A1B4_deg5_exp0 {

//----------------------------------------------------------------------------//

x1b_A1B4_v1x::x1b_A1B4_v1x(std::string mon) {
    if (mon == "ch4") {
        coefficients = std::vector<double> {
 
 6.914183713628242e+00, // 0
 7.039431867882222e-01, // 1
 6.397280045494662e-01, // 2
 7.143159529143733e-01, // 3
-1.450279984562959e+00, // 4
-1.301877447020072e+00, // 5
-7.958310643406495e-02, // 6
-8.115063091379233e-01, // 7
 6.410856058748582e-02, // 8
-8.557225720326274e-01, // 9
-1.899704245667152e+00, // 10
 4.435211531174094e-02, // 11
-8.965251583483286e-01, // 12
 7.481228084470127e-03, // 13
 8.413453365591876e-01, // 14
 1.657947611189469e-01, // 15
-9.124180822910256e-01, // 16
 1.172638283686694e+00, // 17
-8.397344168551895e-01, // 18
-4.192415715240499e-01, // 19
-1.807445569492824e-03, // 20
 4.782360695987890e-02, // 21
 7.638063052427882e-04, // 22
 5.438732084523973e-01, // 23
 1.328292912425294e+00, // 24
 1.112328138950811e+00, // 25
-2.947349483005997e-02, // 26
-2.647761063873971e-02, // 27
-7.314499255552038e-01, // 28
 2.464782630081941e-04, // 29
 5.008679459188946e-04, // 30
-2.863584532285155e-04, // 31
 6.522987800546273e-03, // 32
-3.203793620881907e-02, // 33
 2.185324438832365e-04, // 34
 1.121163540204990e-02, // 35
-1.046940802362836e-03, // 36
 2.071869200563634e+00, // 37
 1.578127187295827e+00, // 38
 6.443807816839333e-01, // 39
 1.386381935164601e-01, // 40
 4.324471700513317e-01, // 41
-2.282810704500523e-01, // 42
 1.782756789233569e-01, // 43
-5.580688926488653e-01, // 44
-4.054948565755861e-05, // 45
-1.108176386435118e-01, // 46
 1.788806175581800e-04, // 47
 2.969072330524916e-02, // 48
-9.623470681030276e-01, // 49
 3.686735846348648e-01, // 50
-2.549523650819612e-04, // 51
-3.263429063871853e-01, // 52
-4.332816223134581e-03, // 53
 1.350613614692035e-02, // 54
-1.864572421161195e+00, // 55
 3.011810951631171e-04, // 56
-5.922755099117846e-05, // 57
-8.505330459142783e-01, // 58
 3.582620876833924e-01, // 59
-1.241625223048353e+00, // 60
-8.045080965933739e-03, // 61
-9.891956746864805e-03, // 62
-1.531057264243375e+00, // 63
 5.415029402512408e-01, // 64
 2.353084322485742e-02, // 65
-4.905750098109656e-04, // 66
-5.393633381445804e-01, // 67
 2.258416449859873e-03, // 68
 9.098768926540463e-01, // 69
 9.657652276787859e-01, // 70
-2.780935948577222e-02, // 71
-7.630953691306434e-01, // 72
-8.195054101813189e-01, // 73
-1.180553580954140e+00, // 74
-1.064563598867319e-01, // 75
-3.557030006388207e-01, // 76
-2.835803413825880e-02, // 77
-9.450909054019764e-01, // 78
 4.215052926446911e-04, // 79
-3.260087697107995e-01, // 80
-2.605982744043365e-02, // 81
 1.560659200242308e-01, // 82
 7.600864293400544e-04, // 83
 3.476561633115353e-03, // 84
 1.249225319974905e-06, // 85
-5.111382398625382e-04, // 86
 8.021798097300144e-07, // 87
 2.568117265025944e-02, // 88
 6.822759596768759e-05, // 89
-5.525386372025131e-02, // 90
 4.385405410228613e-02, // 91
 4.448425409427371e-04, // 92
-9.410639499612241e-05, // 93
-3.289331496490664e-03, // 94
-1.217678539583626e-01, // 95
-2.383171930943966e-05, // 96
 2.413521217577282e-04, // 97
-6.071466776755903e-03, // 98
 1.489910562443420e-07, // 99
 2.405547030713126e-06, // 100
 1.531181114705639e-03, // 101
 2.239631437571822e-06, // 102
 4.673790634381474e-07, // 103
-5.142915053621096e-05, // 104
 3.662375822935582e-03, // 105
-7.158589779338386e-04, // 106
 7.068299163653584e-02, // 107
-3.893565259522595e-03, // 108
-7.042910392200739e-05, // 109
-4.607704691083050e-02, // 110
 4.668364208856403e-01, // 111
 7.331033260549798e-07, // 112
 2.312152346288223e-05, // 113
-1.089773277720457e-05, // 114
 1.037252097709177e-02, // 115
-2.739137619208556e-04, // 116
-2.221474763554481e-03, // 117
-1.289671361974173e-03, // 118
 9.188978964491857e-05, // 119
 2.961802324880276e-05, // 120
-9.657693481270842e-05, // 121
-1.190808926435630e-05, // 122
-5.070129664960805e-01, // 123
 3.257999573159684e-02, // 124
-2.208151819756846e-03, // 125
-1.247318071792302e-06, // 126
 1.609947336001569e-02, // 127
-5.866723133707953e-07, // 128
 9.004536226587241e-05, // 129
 2.541947049348913e-03, // 130
-1.358998930502508e-06, // 131
-5.658088065761403e-01, // 132
 2.094489003486378e-03, // 133
-5.019695784134502e-01, // 134
-3.526958447901748e-03, // 135
 4.144014818963010e-07, // 136
 5.032867450695801e-05, // 137
 5.871860915299803e-03, // 138
 5.933497568472461e-01, // 139
 9.878633117085106e-04, // 140
 2.934335107406904e-01, // 141
 8.040887707347417e-02, // 142
 9.125969167283980e-02, // 143
-3.287172895240080e-03, // 144
-1.190952636220926e-06, // 145
-4.146602486723987e-02, // 146
 1.254329043085791e-04, // 147
-6.924140880956214e-07, // 148
-5.112828753831737e-04, // 149
-1.301282246781800e-02, // 150
 6.027303956582904e-03, // 151
 4.917774781570320e-01, // 152
-4.443221602958519e-03, // 153
-1.256694585039356e-03, // 154
-8.712373244075597e-02, // 155
-3.730119065189077e-02, // 156
-1.712263694769486e-01, // 157
-6.687688007176697e-05, // 158
-4.396297555948074e-05, // 159
 2.850785219572671e-01, // 160
 1.882571344646084e-01, // 161
 1.709269583320334e-05, // 162
 2.868160214908864e-05, // 163
-1.230763528508933e-03, // 164
 1.621520438744679e-07, // 165
 7.382858115607314e-04, // 166
 4.112745068595877e-03, // 167
-2.832117090595519e-02, // 168
 1.896462637554295e-05, // 169
 7.664047754686733e-04, // 170
 2.124011820129472e-01, // 171
-4.746018534730015e-05, // 172
-5.322187239708175e-05, // 173
-3.737122238691511e-02, // 174
 5.379141252574962e-05, // 175
-8.146762618528465e-07, // 176
-5.864975589330074e-02, // 177
-2.187700252547936e-03, // 178
-1.223796461662206e-03, // 179
-3.252098506451394e-02, // 180
-5.075069169034273e-02, // 181
-2.707197320601311e-02, // 182
-1.515976800491137e-01, // 183
 2.633337917828847e-05, // 184
 3.095790295905859e-03, // 185
 6.038723448331831e-01, // 186
 7.247778025598385e-05, // 187
 6.582152695387059e-05, // 188
 4.682846404008734e-02, // 189
-4.699135302876049e-01, // 190
-1.455394781361203e-06, // 191
-1.857646604952004e-01, // 192
 4.416464616741460e-03, // 193
-4.745811957398000e-03, // 194
-4.255090017953875e-02, // 195
 2.249899324801534e-02, // 196
 1.216799112250322e-01, // 197
 1.480145237120888e-03, // 198
 2.362125166198466e-05, // 199
 3.450928988137539e-06, // 200
 3.570100650163137e-01, // 201
-1.656571967540483e-05, // 202
-1.047074927613252e-07, // 203
-4.203061442370540e-03, // 204
-4.152096795280623e-01, // 205
-3.598832082720675e-06}; // 206

   m_k_AB =  9.503712645968341e-01; // A^(-1))
   m_d_AB =  1.800700424600860e+00; // A^(-1))
   m_k_BB =  7.671972591586966e-01; // A^(-1))
   m_d_BB =  6.979931086907743e+00; // A^(-1))
    } // end if (mon == "ch4")
}


std::vector<double> x1b_A1B4_v1x::eval(const double* xyz, const size_t nmon) const
{

    std::vector<double> energies(nmon,0.0);

    for (size_t j = 0; j < nmon; j++) {


        double xcrd[15]; // coordinates of real sites ONLY
    
        std::copy(xyz + j*15, xyz + (j+1)*15, xcrd);
        
        double v[10];
        
        const double* A_1= xcrd + 0;
        const double* B_1= xcrd + 3;
        const double* B_2= xcrd + 6;
        const double* B_3= xcrd + 9;
        const double* B_4= xcrd + 12;
        
        variable vr[10];
        
        v[0]  = vr[0].v_exp0(m_d_AB, m_k_AB, A_1, B_1);
        v[1]  = vr[1].v_exp0(m_d_AB, m_k_AB, A_1, B_2);
        v[2]  = vr[2].v_exp0(m_d_AB, m_k_AB, A_1, B_3);
        v[3]  = vr[3].v_exp0(m_d_AB, m_k_AB, A_1, B_4);
        v[4]  = vr[4].v_exp0(m_d_BB, m_k_BB, B_1, B_2);
        v[5]  = vr[5].v_exp0(m_d_BB, m_k_BB, B_1, B_3);
        v[6]  = vr[6].v_exp0(m_d_BB, m_k_BB, B_1, B_4);
        v[7]  = vr[7].v_exp0(m_d_BB, m_k_BB, B_2, B_3);
        v[8]  = vr[8].v_exp0(m_d_BB, m_k_BB, B_2, B_4);
        v[9]  = vr[9].v_exp0(m_d_BB, m_k_BB, B_3, B_4);
    
         
        
        energies[j] = polynomial::eval(coefficients.data(), v);
        
    }

    return energies;

}

std::vector<double> x1b_A1B4_v1x::eval(const double* xyz,
                double * grad, const size_t nmon, std::vector<double> *virial) const
{

    std::vector<double> energies(nmon,0.0);

    for (size_t j = 0; j < nmon; j++) {

        double xcrd[15]; // coordinates of real sites ONLY
    
        std::copy(xyz + j*15, xyz + (j+1)*15, xcrd);
        
        double v[10];
        
        const double* A_1= xcrd + 0;
        const double* B_1= xcrd + 3;
        const double* B_2= xcrd + 6;
        const double* B_3= xcrd + 9;
        const double* B_4= xcrd + 12;
    
    
        
        variable vr[10];
        
        v[0]  = vr[0].v_exp0(m_d_AB, m_k_AB, A_1, B_1);
        v[1]  = vr[1].v_exp0(m_d_AB, m_k_AB, A_1, B_2);
        v[2]  = vr[2].v_exp0(m_d_AB, m_k_AB, A_1, B_3);
        v[3]  = vr[3].v_exp0(m_d_AB, m_k_AB, A_1, B_4);
        v[4]  = vr[4].v_exp0(m_d_BB, m_k_BB, B_1, B_2);
        v[5]  = vr[5].v_exp0(m_d_BB, m_k_BB, B_1, B_3);
        v[6]  = vr[6].v_exp0(m_d_BB, m_k_BB, B_1, B_4);
        v[7]  = vr[7].v_exp0(m_d_BB, m_k_BB, B_2, B_3);
        v[8]  = vr[8].v_exp0(m_d_BB, m_k_BB, B_2, B_4);
        v[9]  = vr[9].v_exp0(m_d_BB, m_k_BB, B_3, B_4);
    
         
        
        double g[10];
        
        energies[j] = polynomial::eval(coefficients.data(), v, g);
        
        double xgrd[15];
        std::fill(xgrd, xgrd + 15, 0.0);
    
        double* A_1_g = xgrd + 0;
        double* B_1_g = xgrd + 3;
        double* B_2_g = xgrd + 6;
        double* B_3_g = xgrd + 9;
        double* B_4_g = xgrd + 12;
    
    
        vr[0].grads(g[0], A_1_g, B_1_g, A_1, B_1);
        vr[1].grads(g[1], A_1_g, B_2_g, A_1, B_2);
        vr[2].grads(g[2], A_1_g, B_3_g, A_1, B_3);
        vr[3].grads(g[3], A_1_g, B_4_g, A_1, B_4);
        vr[4].grads(g[4], B_1_g, B_2_g, B_1, B_2);
        vr[5].grads(g[5], B_1_g, B_3_g, B_1, B_3);
        vr[6].grads(g[6], B_1_g, B_4_g, B_1, B_4);
        vr[7].grads(g[7], B_2_g, B_3_g, B_2, B_3);
        vr[8].grads(g[8], B_2_g, B_4_g, B_2, B_4);
        vr[9].grads(g[9], B_3_g, B_4_g, B_3, B_4);
    
        for (size_t i = 0; i < 15; i++)
            grad[i + j*15] = xgrd[i];

        if (virial != 0) {
         
            (*virial)[0] += -A_1[0]* A_1_g[0]
                            -B_1[0]* B_1_g[0]
                            -B_2[0]* B_2_g[0]
                            -B_3[0]* B_3_g[0]
                            -B_4[0]* B_4_g[0];
         
            (*virial)[1] += -A_1[0]* A_1_g[1]
                            -B_1[0]* B_1_g[1]
                            -B_2[0]* B_2_g[1]
                            -B_3[0]* B_3_g[1]
                            -B_4[0]* B_4_g[1];
         
            (*virial)[2] += -A_1[0]* A_1_g[2]
                            -B_1[0]* B_1_g[2]
                            -B_2[0]* B_2_g[2]
                            -B_3[0]* B_3_g[2]
                            -B_4[0]* B_4_g[2];
         
            (*virial)[4] += -A_1[1]* A_1_g[1]
                            -B_1[1]* B_1_g[1]
                            -B_2[1]* B_2_g[1]
                            -B_3[1]* B_3_g[1]
                            -B_4[1]* B_4_g[1];
         
            (*virial)[5] += -A_1[1]* A_1_g[2]
                            -B_1[1]* B_1_g[2]
                            -B_2[1]* B_2_g[2]
                            -B_3[1]* B_3_g[2]
                            -B_4[1]* B_4_g[2];
         
            (*virial)[8] += -A_1[2]* A_1_g[2]
                            -B_1[2]* B_1_g[2]
                            -B_2[2]* B_2_g[2]
                            -B_3[2]* B_3_g[2]
                            -B_4[2]* B_4_g[2];
        
            (*virial)[3]=(*virial)[1];
            (*virial)[6]=(*virial)[2];
            (*virial)[7]=(*virial)[5];
        
        } 
 
    }

    return energies;
}

} // namespace x1b_A1B4_deg5_exp0

////////////////////////////////////////////////////////////////////////////////
