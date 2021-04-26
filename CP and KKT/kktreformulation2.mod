param n > 0; # cardinality of the graph
set V ordered := 1..n; #set of vertices

param M {i in V, j in V}; #proximity matrix
#cost parameters:
param quadcostlevel;
param linear_cost1;
param linear_cost2;

param Q1 {i in V, j in V}; #cost 1 matrix
param q1 {i in V}; #cost 1 vector
param Q2_fix {i in V, j in V} default 0;
param q2_fix {i in V} default 0;
param diagQ2 {i in V} default 0;

#variables
var x {i in V} >=0;
var v;
var y {i in V} >=0;
var gamma1;
var gamma2 {i in V}>=0;


#obj. function
minimize obj_fun : v + 0.5*sum{i in V}(sum{j in V}(x[i]*x[j]*Q1[i,j]))+ sum{i in V}(q1[i]*x[i]);

#simplex
s.t. simplex_x: sum{i in V} x[i]=1;
s.t. simplex_y: sum{i in V} y[i]=1;

s.t. nonnegativity: - v <= 0.5*sum{i in V}(sum{j in V}(y[i]*y[j]*Q2_fix[i,j]))+0.5*sum{i in V}(y[i]*y[i]*diagQ2[i]*x[i])+sum{i in V}(q2_fix[i]*y[i])+sum{i in V}(sum{j in V}(x[i]*y[j]*M[j,i]));
s.t. stationarity {i in V}: sum{j in V}(Q2_fix[i,j]*y[j])+diagQ2[i]*x[i]*y[i]+q2_fix[i]+sum{j in V}(M[j,i]*x[j])+ gamma1 - gamma2[i]=0;
s.t. complementarity {i in V}: gamma2[i]*y[i]=0;
