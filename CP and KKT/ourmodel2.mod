param n > 0; # cardinality of the graph
set V ordered := 1..n; #set of vertices

param M {i in V, j in V}; #proximity matrix

#cost parameters:
param Q1 {i in V, j in V}; #cost 1 matrix
param q1 {i in V}; #cost 1 vector
param Q2_fix {i in V, j in V} default 0; #part of cost 2 matrix not dependent on x
param q2_fix {i in V} default 0; #cost 2 vector
param diagQ2 {i in V}; #Q2=Q2_fix+diagonal_matrix where diagonal_matrix[i,i]=diagQ2[i]*x[i] for i in V


param nc >=0; #number of cuts
set CUT := 1..nc; #set of cuts
param y_inner {k in CUT, i in V}; #one vector of dimension n for each "cut k"

#variables
var x {i in V} >=0;
var v;

#obj. function
minimize obj_fun : v + 0.5*sum{i in V}(sum{j in V}(x[i]*x[j]*Q1[i,j]))+ sum{i in V}(q1[i]*x[i]);

#simplex
s.t. simplex_x: sum{i in V} x[i]=1;
#cuts
s.t. cut {k in CUT}: - v <= 0.5*sum{i in V}(sum{j in V}(y_inner[k,i]*y_inner[k,j]*Q2_fix[i,j]))+0.5*sum{i in V}(y_inner[k,i]*y_inner[k,i]*diagQ2[i]*x[i])+sum{i in V}(q2_fix[i]*y_inner[k,i])+sum{i in V}(sum{j in V}(x[i]*y_inner[k,j]*M[j,i]));

#----------------
#inner
param x_star {i in V};

#variables inner
var y {i in V} >=0;

#obj function
minimize obj_inner: 0.5*sum{i in V}(sum{j in V}(y[i]*y[j]*Q2_fix[i,j]))+0.5*sum{i in V}(y[i]*y[i]*diagQ2[i]*x_star[i])+sum{i in V}(q2_fix[i]*y[i])+sum{i in V}(sum{j in V}(x_star[i]*y[j]*M[j,i]));

#constraints
s.t. simplex_y: sum{i in V} y[i]=1;
