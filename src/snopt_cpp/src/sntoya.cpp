#include <ros/ros.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include "snoptProblem.hpp"

using namespace std;

void toyusrfg(int *Status, int *n, double x[],
              int *needF, int *neF, double F[],
              int *needG, int *neG, double G[],
              char *cu, int *lencu,
              int iu[], int *leniu,
              double ru[], int *lenru) {
  if (*needF > 0) {
    F[0] = x[1]; // Objective
    F[1] = x[0] * x[0] + 4 * x[1] * x[1];
    F[2] = (x[0] - 2) * (x[0] - 2) + x[1] * x[1];
  }

  if (*needG > 0) {
    G[0] = 2 * x[0];        // iGfun[0] = 1, jGvar[0] = 0
    G[1] = 8 * x[1];        // iGfun[1] = 1, jGvar[1] = 1
    G[2] = 2 * (x[0] - 2);  // iGfun[2] = 2, jGvar[2] = 0
    G[3] = 2 * x[1];        // iGfun[3] = 2, jGvar[3] = 1
  }
}

int main(int argc, char **argv) {
  // Initialize ROS node
  ros::init(argc, argv, "sntoya_node");
  ros::NodeHandle nh;
  ROS_INFO("SNOPT Toy Node started");

  // Set up SNOPT
  snoptProblemA ToyProb;

  int n = 2;
  int neF = 3;
  int nS = 0, nInf;
  double sInf;

  double *x = new double[n];
  double *xlow = new double[n];
  double *xupp = new double[n];
  double *xmul = new double[n];
  int *xstate = new int[n];

  double *F = new double[neF];
  double *Flow = new double[neF];
  double *Fupp = new double[neF];
  double *Fmul = new double[neF];
  int *Fstate = new int[neF];

  int ObjRow = 0;
  double ObjAdd = 0;

  int Cold = 0;

  // Set bounds
  xlow[0] = 0.0; xlow[1] = -1e20;
  xupp[0] = 1e20; xupp[1] = 1e20;
  xstate[0] = 0; xstate[1] = 0;

  Flow[0] = -1e20; Flow[1] = -1e20; Flow[2] = -1e20;
  Fupp[0] = 1e20; Fupp[1] = 4.0; Fupp[2] = 5.0;
  Fmul[0] = 0; Fmul[1] = 0; Fmul[2] = 0;
  x[0] = 1.0;
  x[1] = 1.0;

  // Set up Jacobian
  int lenA = 6;
  int *iAfun = new int[lenA];
  int *jAvar = new int[lenA];
  double *A = new double[lenA];

  int lenG = 6;
  int *iGfun = new int[lenG];
  int *jGvar = new int[lenG];

  int neA = 1, neG = 4;
  iGfun[0] = 1; jGvar[0] = 0;
  iGfun[1] = 1; jGvar[1] = 1;
  iGfun[2] = 2; jGvar[2] = 0;
  iGfun[3] = 2; jGvar[3] = 1;

  iAfun[0] = 0; jAvar[0] = 1; A[0] = 1.0;

  // Initialize SNOPT
  ToyProb.initialize("", 0);
  ToyProb.setProbName("Toy");
  ToyProb.setPrintFile("Toy.out");
  ToyProb.setSpecsFile("/home/dat/catkin_ws/src/snopt_cpp/config/sntoya.spc");
  ToyProb.setIntParameter("Derivative option", 1);
  ToyProb.setIntParameter("Major Iteration limit", 250);
  ToyProb.setIntParameter("Verify level", 3);

  // Solve the problem
  ToyProb.solve(Cold, neF, n, ObjAdd, ObjRow, toyusrfg,
                iAfun, jAvar, A, neA,
                iGfun, jGvar, neG,
                xlow, xupp, Flow, Fupp,
                x, xstate, xmul,
                F, Fstate, Fmul,
                nS, nInf, sInf);

  ROS_INFO("Optimized variables:");
  for (int i = 0; i < n; i++) {
    ROS_INFO("x[%d] = %f", i, x[i]);
  }
  if (nInf == 0 && sInf == 0) {
    ROS_INFO("Optimization successful");
  } else {
    ROS_INFO("Optimization failed: nInf = %d, sInf = %f", nInf, sInf);
  }

  // Clean up
  delete[] iAfun; delete[] jAvar; delete[] A;
  delete[] iGfun; delete[] jGvar;
  delete[] x; delete[] xlow; delete[] xupp;
  delete[] xmul; delete[] xstate;
  delete[] F; delete[] Flow; delete[] Fupp;
  delete[] Fmul; delete[] Fstate;

  // Keep node running
  ros::spin();

  return 0;
}