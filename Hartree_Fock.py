import numpy as np
import math

R = 1.4
ZetaH = 1.24
ZetaHe = 2.0925
ZH = 1
ZHe = 2
Coef = [0.444635, 0.535328, 0.154329] #Index starts from 0
Hexp = [0.168856, 0.623913, 3.42525]
Heexp = [0.480844, 1.776691, 9.753934]

N = 3
D_H = np.zeros(3)
D_He = np.zeros(3)
A = np.zeros((2,2))
B = np.zeros((2,2))
C = np.zeros((2,2))
Eigen = np.zeros((2,2))
H = np.zeros((2,2))
S_mat = np.zeros((2,2))
X = np.zeros((2,2))
XT = np.zeros((2,2))
G = np.zeros((2,2))
P = np.zeros((2,2))
oldP = np.zeros((2,2))
F = np.zeros((2,2))
FPRIME = np.zeros((2,2))
CPRIME = np.zeros((2,2))
Di = np.zeros((2,2))
E = ENTot = 0
TT = np.zeros((2,2,2,2))
s12 = T11 = T12 = T22 = 0
V11A = V22A = V11B = V22B = 0
V12A = V12B = 0
V1111 = V2111 = V2121 = V2211 = V2221 = V2222 = 0
diffeng = 0

def initialize():
    global s12, T11, T12, T22
    global V11A, V12A, V22A, V11B, V12B, V22B
    global ZH, ZHe
    global V1111, V2111, V2121, V2211, V2221, V2222
    global H, G, F, XT, FPRIME, CPRIME, P, C
    global D_H, D_He
    global A, B, C, Eigen, S_mat, X, oldP, E, ENTot, TT, diffeng
    D_H = np.zeros(3)
    D_He = np.zeros(3)
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))
    C = np.zeros((2, 2))
    Eigen = np.zeros((2, 2))
    H = np.zeros((2, 2))
    S_mat = np.zeros((2, 2))
    X = np.zeros((2, 2))
    XT = np.zeros((2, 2))
    G = np.zeros((2, 2))
    P = np.zeros((2, 2))
    oldP = np.zeros((2, 2))
    F = np.zeros((2, 2))
    FPRIME = np.zeros((2, 2))
    CPRIME = np.zeros((2, 2))
    E = ENTot = 0
    TT = np.zeros((2, 2, 2, 2))
    s12 = T11 = T12 = T22 = 0
    V11A = V22A = V11B = V22B = 0
    V12A = V12B = 0
    V1111 = V2111 = V2121 = V2211 = V2221 = V2222 = 0
    diffeng = 0

def D(N):
    for i in range(N):
        D_H[i] = Coef[i]*pow(2*Hexp[i]/np.pi, 0.75)
        D_He[i] = Coef[i]*pow(2*Heexp[i]/np.pi, 0.75)

def F0(arg):
    if arg>pow(10, -6):
        return np.sqrt(np.pi/arg)*math.erf(np.sqrt(arg))/2
    else:
        return 1-arg/3

def intgrl(N, R):
    global s12, T11, T12, T22
    global V11A, V12A, V22A, V11B, V12B, V22B
    global ZH, ZHe
    global V1111, V2111, V2121, V2211, V2221, V2222
    for i in range(N):
        for j in range(N):
            s12 = s12 + S(Hexp[i], Heexp[j], R*R) * D_He[i] * D_H[j]
            T11 = T11 + T(Heexp[i], Heexp[j], 0)*D_He[i]*D_He[j]
            T12 = T12 + T(Heexp[i], Hexp[j], R*R)*D_He[i]*D_H[j]
            T22 = T22 + T(Hexp[i], Hexp[j], 0) * D_H[i] * D_H[j]
            Rap = Hexp[j]*R/(Heexp[i]+Hexp[j])
            Rbp = R - Rap
            V11A = V11A + V(Heexp[i], Heexp[j], 0, 0, ZHe)*D_He[i]*D_He[j]
            V12A = V12A + V(Heexp[i], Hexp[j], R*R, Rap*Rap, ZHe) * D_He[i] * D_H[j]
            V22A = V22A + V(Hexp[i], Hexp[j], 0, R*R, ZHe) * D_H[i] * D_H[j]
            V11B = V11B + V(Heexp[i], Heexp[j], 0, R*R, ZH) * D_He[i] * D_He[j]
            V12B = V12B + V(Heexp[i], Hexp[j], R*R, Rbp*Rbp, ZH) * D_He[i] * D_H[j]
            V22B = V22B + V(Hexp[i], Hexp[j], 0, 0, ZH) * D_H[i] * D_H[j]

    for i in range(N):
        for j in range(N):
            for k in range(N):
                for l in range(N):
                    Rap = Hexp[i] * R / (Hexp[i] + Heexp[j])
                    Rbp = R - Rap
                    Raq = Hexp[k]*R/(Hexp[k]+Heexp[l])
                    Rbq = R - Raq
                    Rpq = Rap - Raq
                    V1111 = V1111 + twoeint(Heexp[i], Heexp[j], Heexp[k], Heexp[l], 0, 0, 0)*D_He[i]*D_He[j]*D_He[k]*D_He[l]
                    V2111 = V2111 + twoeint(Hexp[i], Heexp[j], Heexp[k], Heexp[l], R*R, 0, Rap*Rap)*D_H[i] * D_He[j] * D_He[k] * D_He[l]
                    V2121 = V2121 + twoeint(Hexp[i], Heexp[j], Hexp[k], Heexp[l], R*R, R*R, Rpq*Rpq) * D_He[i] * D_H[j] * D_He[k] * D_H[l]
                    V2211 = V2211 + twoeint(Hexp[i], Hexp[j], Heexp[k], Heexp[l], 0, 0, R*R) * D_He[i] * D_He[j] * D_H[k] * D_H[l]
                    V2221 = V2221 + twoeint(Hexp[i], Hexp[j], Hexp[k], Heexp[l], 0, R*R, Rbq*Rbq) * D_H[i] * D_H[j] * D_H[k] * D_He[l]
                    V2222 = V2222 + twoeint(Hexp[i], Hexp[j], Hexp[k], Hexp[l], 0, 0, 0) * D_H[i] * D_H[j] * D_H[k] * D_H[l]

def collect():
    H[0][0] = T11 + V11A + V11B
    H[0][1] = T12 + V12A + V12B
    H[1][0] = H[0][1]
    H[1][1] = T22 + V22A + V22B
    S_mat[0][0] = 1.0
    S_mat[0][1] = s12
    S_mat[1][0] = S_mat[0][1]
    S_mat[1][1] = 1.0
    X[0][0] = 1.0e0 / np.sqrt(2.0 * (1.0 + s12))
    X[1][0] = X[0][0]
    X[0][1] = 1.0e0 / np.sqrt(2.0 * (1.0 - s12))
    X[1][1] = -X[0][1]
    XT[0][0] = X[0][0]
    XT[0][1] = X[1][0]
    XT[1][0] = X[0][1]
    XT[1][1] = X[1][1]
    TT[0][0][0][0] = V1111
    TT[1][0][0][0] = V2111
    TT[0][1][0][0] = V2111
    TT[0][0][1][0] = V2111
    TT[0][0][0][1] = V2111
    TT[1][0][1][0] = V2121
    TT[0][1][1][0] = V2121
    TT[1][0][0][1] = V2121
    TT[0][1][0][1] = V2121
    TT[1][1][0][0] = V2211
    TT[0][0][1][1] = V2211
    TT[1][1][1][0] = V2221
    TT[1][1][0][1] = V2221
    TT[1][0][1][1] = V2221
    TT[0][1][1][1] = V2221
    TT[1][1][1][1] = V2222
    #print(H)

def twoeint(a,b,c,d,Rab2,Rcd2,Rpq2):
    term1 = 2*pow(np.pi, 2.5)/((a+b)*(c+d)*pow(a+b+c+d, 1/2))
    term2 = np.exp(-1*a*b*Rab2/(a+b)-c*d*Rcd2/(c+d))
    term3 = (a+b)*(c+d)*Rpq2/(a+b+c+d)
    return term1*term2*F0(term3)

def dipole(a,b,R):
    return R*pow(np.pi/(a+b), 1.5)*np.exp(-1*a*b*R*R/(a+b))

def dipoleintegral():
    global Di
    for i in range(N):
        for j in range(N):
            Di[0][0] = Di[0][0] + dipole(Heexp[i], Heexp[j], 0) * D_He[i] * D_He[j]
            Di[0][1] = Di[0][1] + dipole(Heexp[i], Hexp[j], R) * D_He[i] * D_H[j]
            Di[1][0] = Di[1][0] + dipole(Hexp[i], Heexp[j], 0) * D_H[i] * D_He[j]
            Di[1][1] = Di[1][1] + dipole(Hexp[i], Hexp[j], R) * D_H[i] * D_H[j]

def S(a,b,Rab2):
    return pow(np.pi/(a+b), 1.5)*np.exp(-1*a*b*Rab2/(a+b))

def T(a,b,Rab2):
    return (a*b)/(a+b)*(3.0-2.0*a*b*Rab2/(a+b))*pow(np.pi/(a+b),1.5)*np.exp(-a*b*Rab2/(a+b))

def V(a,b,Rab2,Rcp2,ZC):
    term1 = 2*np.pi/(a+b)*F0((a+b)*Rcp2)*np.exp(-1*a*b*Rab2/(a+b))
    return term1*(-1)*ZC

def diag(A, B, Eigen):
    if (abs(A[0][0] - A[1][1]) > pow(10,-20)):
        THETA = 0.5 * np.arctan(2.0 * A[0][1] / (A[0][0] - A[1][1]))
    else:
        THETA = np.pi / 4.0
    B[0][0] = np.cos(THETA)
    B[1][0] = np.sin(THETA)
    B[0][1] = np.sin(THETA)
    B[1][1] = -1*np.cos(THETA)
    Eigen[0][0] = A[0][0] * pow(np.cos(THETA), 2) + A[1][1] * pow(np.sin(THETA), 2) + A[0][1] * np.sin(2.0 * THETA);
    Eigen[1][1] = A[1][1] * pow(np.cos(THETA), 2) + A[0][0] * pow(np.sin(THETA), 2) - A[0][1] * np.sin(2.0 * THETA);
    Eigen[1][0] = 0.0
    Eigen[0][1] = 0.0
    if (Eigen[1][1] < Eigen[0][0]):
        TEMP = Eigen[1][1]
        Eigen[1][1] = Eigen[0][0]
        Eigen[0][0] = TEMP
        TEMP = B[0][1]
        B[0][1] = B[0][0]
        B[0][0] = TEMP
        TEMP = B[1][1]
        B[1][1] = B[1][0]
        B[1][0] = TEMP
    return B, E

def formG():
    global G, P, TT
    G = np.zeros((2,2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    G[i][j] = G[i][j] + P[k][l] * (TT[i][j][k][l] - 0.5 * TT[i][l][k][j])

def SCF(R):
    criteria = pow(10, -4)
    currit = 0
    maxitg = 100
    diffeng = 1000
    ENTot = EN = 0
    global H, G, F, XT, FPRIME, CPRIME, P, C
    while diffeng > criteria and currit < maxitg:
        currit = currit + 1
        print("\n=========================")
        print("**** Iteration "+str(currit)+" ****")
        formG()
        print("The G Array\n", G)
        F = H + G
        print("The F Array\n", F)
        EN = 0
        for i in range(2):
            for j in range(2):
                EN = EN + 0.5*P[i][j]*(H[i][j]+F[i][j])
        print("=> Electronic energy: "+str(EN))
        G = F.dot(X)
        FPRIME = XT.dot(G)
        print("The F\' Array\n", FPRIME)
        diag(FPRIME, CPRIME, Eigen)
        print("The C\' Array\n", CPRIME)
        print("The Eigen Array\n", Eigen)
        C = X.dot(CPRIME)
        print("The C Array\n", C)
        oldP = P
        P = np.zeros((2,2))
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    P[i][j] = P[i][j] + 2*C[i][k]*C[j][k]
        print("The P Array\n", P)
        diffeng = 0
        for i in range(2):
            for j in range(2):
                diffeng = diffeng + pow(P[i][j]-oldP[i][j], 2)
        diffeng = np.sqrt(diffeng/4)
        print("Delta: " ,diffeng)
        if diffeng < criteria:
            ENTot = EN + ZH * ZHe / R
            print("\n=============================")
            print("Calculation Converged! :D")
            print("Electronic energy: ",EN)
            print("Total energy: " ,ENTot)
            print("=============================")
    return EN, ENTot

def main(R):
    print("Hartree-Fock for STO-3G, HeH+")
    D(N)
    dipoleintegral()
    intgrl(N, R)
    collect()
    x, y = SCF(R)
    print("\nMulliken Populations:\n", P.dot(S_mat))
    print("\nDiople moment matrix\n", -1*P.dot(Di))
    initialize()
    return x, y


main(1.4)

# Loop for the geometry scan
#for item in [1.0, 1.1, 1.2, 1.3, 1.4, 2.0]:
#    print("When R = \'",item,"\'" )
#    print(item, main(item),"\n")