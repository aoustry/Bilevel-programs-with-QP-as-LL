param n > 0; #dimensions of features vector w/y
set N ordered := 1..n; #set of dimensions of w/y

param p_max > 0; #number of data in the dataset
set P ordered :=1..p_max; #indeces of the dataset

param r_dim >= 0; #number of constraints
set R ordered := 1..r_dim;

#param a {r in R, i in N} default 0;
#param b {r in R} default 0;

#ref
param epsilon {p in P};
param Q_ref {i in N, j in N};
param q_ref {i in N};
param c_ref;

param z {p in P}; #outputs
param w {p in P, i in N}; #feature vectors

#variables
var Q {i in N, j in N};
var q {i in N};
var c;
var y {i in N}>=0, <=1;
var lambda1 {i in N} >=0;
var lambda2 {i in N} >=0;

#obj. function (least-squares error)
minimize LSE : sum{p in P} (z[p]-sum{i in N}(0.5*sum{j in N}(w[p,i]*w[p,j]*Q[i,j])) - sum{i in N}(q[i]*w[p,i])-c)^2;

#constraints
s.t. symmetry {i in N, j in N: i<j}: Q[i,j] = Q[j,i];
s.t. nonnegativity: sum{i in N}(0.5*sum{j in N}(y[i]*y[j]*Q[i,j])) + sum{i in N}(q[i]*y[i])+c >= 0;
s.t. stationarity {i in N}: sum{j in N}(Q[i,j]*y[j])+q[i] - lambda1[i] +lambda2[i]=0;
s.t. complementary1 {i in N}: lambda1[i]*y[i] = 0;
s.t. complementary2 {i in N}: lambda2[i]*(y[i]-1) = 0;
 

