/******************************************************************************
Copyright 2019 The Regents of the University of California.
All Rights Reserved.

Permission to copy, modify and distribute any part of this Software for
educational, research and non-profit purposes, without fee, and without
a written agreement is hereby granted, provided that the above copyright
notice, this paragraph and the following three paragraphs appear in all
copies.

Those desiring to incorporate this Software into commercial products or
use for commercial purposes should contact the:
Office of Innovation & Commercialization
University of California, San Diego
9500 Gilman Drive, Mail Code 0910
La Jolla, CA 92093-0910
Ph: (858) 534-5815
FAX: (858) 534-7345
E-MAIL: invent@ucsd.edu

IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR
DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE, EVEN IF THE UNIVERSITY
OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THE SOFTWARE PROVIDED HEREIN IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF
CALIFORNIA HAS NO OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES,
ENHANCEMENTS, OR MODIFICATIONS. THE UNIVERSITY OF CALIFORNIA MAKES NO
REPRESENTATIONS AND EXTENDS NO WARRANTIES OF ANY KIND, EITHER IMPLIED OR
EXPRESS, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE
SOFTWARE WILL NOT INFRINGE ANY PATENT, TRADEMARK OR OTHER RIGHTS.
******************************************************************************/

#ifndef UNITTESTS_SETUP_MONMIX_H
#define UNITTESTS_SETUP_MONMIX_H

#define SETUP_MONMIX                                                                                                  \
    std::vector<std::string> atom_names{"C", "H",  "H",  "H", "H",  "O",  "H", "H",  "C",  "O", "O", "X", "He", "Ar", \
                                        "F", "Cl", "Br", "I", "li", "Na", "K", "Rb", "Cs", "H", "H", "H", "H"};       \
    std::vector<std::string> monomer_names = {"ch4", "h2o", "co2", "dummy", "he", "ar", "f",  "cl",                   \
                                              "br",  "i",   "li",  "na",    "k",  "rb", "cs", "h4_dummy"};            \
    const int n_monomers = monomer_names.size();                                                                      \
    std::vector<size_t> islocal = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};                                   \
    std::vector<size_t> n_atoms_vector = {5, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4};                            \
    std::vector<size_t> n_sites_vector = {5, 4, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4};                            \
    std::vector<size_t> first_index = {0, 5, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};                  \
    std::vector<size_t> first_index_realSites = {0, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};        \
    const size_t n_atoms = 27;                                                                                        \
    const size_t n_sites = 28;                                                                                        \
    std::vector<double> coords{                                                                                       \
        -3.1286212200e+00, 2.6245654300e+00,  0.0000000000e+00,  -2.7719668000e+00, 1.6157554300e+00,                 \
        0.0000000000e+00,  -2.7719483800e+00, 3.1289636200e+00,  8.7365150000e-01,  -2.7719483800e+00,                \
        3.1289636200e+00,  -8.7365150000e-01, -4.1986212200e+00, 2.6245786100e+00,  0.0000000000e+00,                 \
        2.6419468100e+00,  2.0915411000e+00,  0.0000000000e+00,  3.6019468100e+00,  2.0915411000e+00,                 \
        0.0000000000e+00,  2.3214922200e+00,  2.9964769400e+00,  0.0000000000e+00,  2.7783960239e+00,                 \
        2.2846122753e+00,  0.0000000000e+00,  -1.8539978000e-01, -1.3615295300e+00, -0.0000000000e+00,                \
        -1.4437997800e+00, -1.3615295300e+00, 0.0000000000e+00,  1.0730002200e+00,  -1.3615295300e+00,                \
        0.0000000000e+00,  -3.8959445700e+00, -3.5631517400e+00, 0.0000000000e+00,  -1.4327778000e-01,                \
        4.2420790900e+00,  3.2471167200e+00,  4.0764742000e+00,  -2.0748484900e+00, 1.2772387600e+00,                 \
        1.6733565200e+00,  -4.6454177000e-01, 4.1777756100e+00,  -1.6752280000e-01, 8.0763264000e-01,                 \
        -4.6986040300e+00, -1.0810387000e+00, 5.8842208600e+00,  -3.6893933200e+00, -1.9107993400e+00,                \
        -1.6131390000e+00, 3.6955216900e+00,  1.6061109400e+00,  1.8032622300e+00,  8.1120853100e+00,                 \
        4.4863439300e+00,  5.8238642400e+00,  3.4167310300e+00,  4.5569804100e+00,  6.5013788800e+00,                 \
        -5.9791922000e-01, 3.2197436700e+00,  6.2122316800e+00,  -9.1116127900e+00, -3.6032914900e+00,                \
        -2.2792386800e+00, -4.8301118200e+00, 0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+01,                 \
        1.0000000000e+00,  0.0000000000e+00,  1.0000000000e+01,  1.0000000000e+00,  1.0000000000e+00,                 \
        1.0000000000e+01,  0.0000000000e+00,  1.0000000000e+00,  1.0000000000e+01};                                   \
    std::vector<double> real_coords{                                                                                  \
        -3.1286212200e+00, 2.6245654300e+00,  0.0000000000e+00,  -2.7719668000e+00, 1.6157554300e+00,                 \
        0.0000000000e+00,  -2.7719483800e+00, 3.1289636200e+00,  8.7365150000e-01,  -2.7719483800e+00,                \
        3.1289636200e+00,  -8.7365150000e-01, -4.1986212200e+00, 2.6245786100e+00,  0.0000000000e+00,                 \
        2.6419468100e+00,  2.0915411000e+00,  0.0000000000e+00,  3.6019468100e+00,  2.0915411000e+00,                 \
        0.0000000000e+00,  2.3214922200e+00,  2.9964769400e+00,  0.0000000000e+00,  0.0000000000e+00,                 \
        0.0000000000e+00,  0.0000000000e+00,  -1.8539978000e-01, -1.3615295300e+00, -0.0000000000e+00,                \
        -1.4437997800e+00, -1.3615295300e+00, 0.0000000000e+00,  1.0730002200e+00,  -1.3615295300e+00,                \
        0.0000000000e+00,  -3.8959445700e+00, -3.5631517400e+00, 0.0000000000e+00,  -1.4327778000e-01,                \
        4.2420790900e+00,  3.2471167200e+00,  4.0764742000e+00,  -2.0748484900e+00, 1.2772387600e+00,                 \
        1.6733565200e+00,  -4.6454177000e-01, 4.1777756100e+00,  -1.6752280000e-01, 8.0763264000e-01,                 \
        -4.6986040300e+00, -1.0810387000e+00, 5.8842208600e+00,  -3.6893933200e+00, -1.9107993400e+00,                \
        -1.6131390000e+00, 3.6955216900e+00,  1.6061109400e+00,  1.8032622300e+00,  8.1120853100e+00,                 \
        4.4863439300e+00,  5.8238642400e+00,  3.4167310300e+00,  4.5569804100e+00,  6.5013788800e+00,                 \
        -5.9791922000e-01, 3.2197436700e+00,  6.2122316800e+00,  -9.1116127900e+00, -3.6032914900e+00,                \
        -2.2792386800e+00, -4.8301118200e+00, 0.0000000000e+00,  0.0000000000e+00,  1.0000000000e+01,                 \
        1.0000000000e+00,  0.0000000000e+00,  1.0000000000e+01,  1.0000000000e+00,  1.0000000000e+00,                 \
        1.0000000000e+01,  0.0000000000e+00,  1.0000000000e+00,  1.0000000000e+01};                                   \
    std::vector<double> polfac{                                                                                       \
        1.3932677000e+00, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 1.3100000000e+00,   \
        2.9400000000e-01, 2.9400000000e-01, 1.3100000000e+00, 1.4716770000e+00, 7.6979000000e-01, 7.6979000000e-01,   \
        0.0000000000e+00, 2.0493754000e-01, 1.6450000000e+00, 2.4669000000e+00, 5.3602000000e+00, 7.1668000000e+00,   \
        1.0118400000e+01, 2.8500000000e-02, 1.4760000000e-01, 8.1840000000e-01, 1.3614000000e+00, 2.3660000000e+00,   \
        0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00};                                      \
    std::vector<double> pol{1.3932677000e+00, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, \
                            1.3100000000e+00, 2.9400000000e-01, 2.9400000000e-01, 0.0000000000e+00, 1.4716770000e+00, \
                            7.6979000000e-01, 7.6979000000e-01, 0.0000000000e+00, 2.0493754000e-01, 1.6450000000e+00, \
                            2.4669000000e+00, 5.3602000000e+00, 7.1668000000e+00, 1.0118400000e+01, 2.8500000000e-02, \
                            1.4760000000e-01, 8.1840000000e-01, 1.3614000000e+00, 2.3660000000e+00, 0.0000000000e+00, \
                            0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00};                                    \
    std::vector<double> charges{-5.3857300000e-01, 1.3464325000e-01,  1.3464325000e-01,  1.3464325000e-01,            \
                                1.3464325000e-01,  0.0000000000e+00,  5.8639591799e-01,  5.8639591692e-01,            \
                                -1.1727918349e+00, 7.0602700000e-01,  -3.5301350000e-01, -3.5301350000e-01,           \
                                0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,  -1.0000000000e+00,           \
                                -1.0000000000e+00, -1.0000000000e+00, -1.0000000000e+00, 1.0000000000e+00,            \
                                1.0000000000e+00,  1.0000000000e+00,  1.0000000000e+00,  1.0000000000e+00,            \
                                0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00};           \
    std::vector<double> real_polfac{                                                                                  \
        1.3932677000e+00, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 1.3100000000e+00,   \
        2.9400000000e-01, 2.9400000000e-01, 1.4716770000e+00, 7.6979000000e-01, 7.6979000000e-01, 0.0000000000e+00,   \
        2.0493754000e-01, 1.6450000000e+00, 2.4669000000e+00, 5.3602000000e+00, 7.1668000000e+00, 1.0118400000e+01,   \
        2.8500000000e-02, 1.4760000000e-01, 8.1840000000e-01, 1.3614000000e+00, 2.3660000000e+00, 0.0000000000e+00,   \
        0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00};                                                        \
    std::vector<double> real_pol{                                                                                     \
        1.3932677000e+00, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 3.8978363000e-01, 1.3100000000e+00,   \
        2.9400000000e-01, 2.9400000000e-01, 1.4716770000e+00, 7.6979000000e-01, 7.6979000000e-01, 0.0000000000e+00,   \
        2.0493754000e-01, 1.6450000000e+00, 2.4669000000e+00, 5.3602000000e+00, 7.1668000000e+00, 1.0118400000e+01,   \
        2.8500000000e-02, 1.4760000000e-01, 8.1840000000e-01, 1.3614000000e+00, 2.3660000000e+00, 0.0000000000e+00,   \
        0.0000000000e+00, 0.0000000000e+00, 0.0000000000e+00};                                                        \
    std::vector<double> real_charges{-5.3857300000e-01, 1.3464325000e-01,  1.3464325000e-01,  1.3464325000e-01,       \
                                     1.3464325000e-01,  0.0000000000e+00,  5.8639591799e-01,  5.8639591692e-01,       \
                                     7.0602700000e-01,  -3.5301350000e-01, -3.5301350000e-01, 0.0000000000e+00,       \
                                     0.0000000000e+00,  0.0000000000e+00,  -1.0000000000e+00, -1.0000000000e+00,      \
                                     -1.0000000000e+00, -1.0000000000e+00, 1.0000000000e+00,  1.0000000000e+00,       \
                                     1.0000000000e+00,  1.0000000000e+00,  1.0000000000e+00,  0.0000000000e+00,       \
                                     0.0000000000e+00,  0.0000000000e+00,  0.0000000000e+00};                         \
    std::vector<std::pair<std::string, size_t>> internal_mon_type_count{                                              \
        {"ch4", 1}, {"h2o", 1}, {"co2", 1}, {"dummy", 1}, {"he", 1}, {"ar", 1}, {"f", 1},  {"cl", 1},                 \
        {"br", 1},  {"i", 1},   {"li", 1},  {"na", 1},    {"k", 1},  {"rb", 1}, {"cs", 1}, {"h4_dummy", 1}};          \
    std::vector<size_t> internal_original_to_current_order{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};     \
    std::vector<std::pair<size_t, size_t>> internal_orginal_order{                                                    \
        {0, 0},  {1, 5},  {2, 9},   {3, 12},  {4, 13},  {5, 14},  {6, 15},  {7, 16},                                  \
        {8, 17}, {9, 18}, {10, 19}, {11, 20}, {12, 21}, {13, 22}, {14, 23}, {15, 24}};                                \
    std::vector<std::pair<size_t, size_t>> internal_original_order_realSites{                                         \
        {0, 0},  {1, 5},  {2, 8},   {3, 11},  {4, 12},  {5, 13},  {6, 14},  {7, 15},                                  \
        {8, 16}, {9, 17}, {10, 18}, {11, 19}, {12, 20}, {13, 21}, {14, 22}, {15, 23}};                                \
    std::vector<double> C6_long_range{17.41398863,                                                                    \
                                      6.064748037,                                                                    \
                                      6.064748037,                                                                    \
                                      6.064748037,                                                                    \
                                      6.064748037,                                                                    \
                                      15.40523357222455098728,                                                        \
                                      4.48258697649551357815,                                                         \
                                      4.48258697649551357815,                                                         \
                                      17.91673320223304547491,                                                        \
                                      13.04205731316957524126,                                                        \
                                      13.04205731316957524126,                                                        \
                                      0.00000,                                                                        \
                                      4.93437037524,                                                                  \
                                      43.09834,                                                                       \
                                      25.56412750183350184739,                                                        \
                                      57.88297168036554772821,                                                        \
                                      74.56169774397084024344,                                                        \
                                      105.39445721563933883337,                                                       \
                                      3.24887148714749872914,                                                         \
                                      16.02787872333703428437,                                                        \
                                      37.63136349992751547203,                                                        \
                                      49.17633137941422098718,                                                        \
                                      65.76255818916154320248,                                                        \
                                      0.00000,                                                                        \
                                      0.00000,                                                                        \
                                      0.00000,                                                                        \
                                      0.00000};                                                                       \
    std::vector<std::string> internal_monomer_names{"ch4", "h2o", "co2", "dummy", "he", "ar", "f",  "cl",             \
                                                    "br",  "i",   "li",  "na",    "k",  "rb", "cs", "h4_dummy"};
#endif
