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


param nc >=0; #number of cuts
set CUT := 1..nc; #set of cuts
param y_inner {k in CUT, i in V}; #one vector of dimension n for each "cut k"

#variables
var x {i in V} >=0;
var v;
var Q2 {i in V, j in V}; #cost 2 matrix (will depend on x)
var q2 {i in V}; #cost 2 vector (will depend on x)


#obj. function
minimize obj_fun : v;

#simplex
s.t. simplex_x: sum{i in V} x[i]=1;
#links
s.t. matrix {i in V, j in V}: Q2[i,j]=Q2_fix[i,j];
s.t. vector {i in V}: q2[i]=q2_fix[i];
#s.t. matrix {i in V, j in V: i=j}: Q2[i,j]=3*x[i];
#s.t. matrix_const {i in V, j in V: i != j}: Q2[i,j]=Q2_fix[i,j];
#s.t. vector {i in V: i=2 or i=5}: q2[i]=x[i];
#s.t. vector_const {i in V: i!=2 and i!=5}: q2[i]=q2_fix[i];
#cuts
s.t. cut {k in CUT}: - v + 0.5*sum{i in V}(sum{j in V}(x[i]*x[j]*Q1[i,j]))+ sum{i in V}(q1[i]*x[i]) <= 0.5*sum{i in V}(sum{j in V}(y_inner[k,i]*y_inner[k,j]*Q2[i,j]))+sum{i in V}(q2[i]*y_inner[k,i])+sum{i in V}(sum{j in V}(x[i]*y_inner[k,j]*M[i,j]));

#----------------
#inner

param Q2_star {i in V, j in V};
param q2_star {i in V};
param x_star {i in V};


#variables inner
var y {i in V} >=0;

#obj function
minimize obj_inner: 0.5*sum{i in V}(sum{j in V}(y[i]*y[j]*Q2_star[i,j]))+sum{i in V}(q2_star[i]*y[i])+sum{i in V}(sum{j in V}(x_star[i]*y[j]*M[i,j]));

#constraints
s.t. simplex_y: sum{i in V} y[i]=1;
