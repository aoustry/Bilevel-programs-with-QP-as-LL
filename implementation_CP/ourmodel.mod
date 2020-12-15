param n > 0; #dimensions of features vector w/y
set N ordered := 1..n; #set of dimensions of w/y

param p_max > 0; #number of data in the dataset
set P ordered :=1..p_max; #indeces of the dataset

#ref
param epsilon {p in P};
param Q_ref {i in N, j in N};
param q_ref {i in N};
param c_ref;

param z {p in P}; #outputs
param w {p in P, i in N}; #feature vectors

param nc >=0; #number of cuts
set CUT := 1..nc; #set of cuts
param y_inner {k in CUT, i in N}; #one vector of dimension n for each "cut k"

#variables
var Q {i in N, j in N};
var q {i in N};
var c;

#obj. function (least-squares error)
minimize LSE : sum{p in P} (z[p]-(sum{i in N}(0.5*sum{j in N}(w[p,i]*w[p,j]*Q[i,j]) + q[i]*w[p,i]))-c)^2;

#constraints
s.t. symmetry {i in N, j in N: i<j}: Q[i,j] = Q[j,i];

#cuts
s.t. cut {k in CUT}: sum{i in N}(0.5*sum{j in N}(y_inner[k,i]*y_inner[k,j]*Q[i,j]) + q[i]*y_inner[k,i])+c >= 0;

#----------------
#inner
param r_dim > 0; #number of constraints
set R ordered := 1..r_dim;
param Q_star {i in N, j in N}; 
param q_star {i in N};
param c_star;
param a {r in R, i in N};
param b {r in R};

#variables inner
var y {i in N};

#obj function
minimize obj_inner: sum{i in N}(0.5*sum{j in N}(y[i]*y[j]*Q_star[i,j]) + q_star[i]*y[i])+c_star;

#constraints
s.t. polytope {r in R}: sum{i in N} a[r,i]*y[i] <= b[r];
