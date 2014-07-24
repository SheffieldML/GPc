GPc
===

Gaussian process code in C++ including some old implementations of GPs, GP-LVM and IVM. The software dates back to 2005 and was modified regularly until around 2007. The combined release was moved from SVN to github in June 2014. The documentation below is still mainly from 2007.

<h3>Design Philosophy</h3>

The software is written in C++ to try and get a degree of flexibility in the models that can be used without a serious performance hit. 

The software is mainly written in C++ but relies for some functions on FORTRAN code by other authors and the LAPACK and BLAS libraries. 

<h2>Compiling the Software</h2>

The software was written with gcc vs 3.2.2. There are definitely Standard Template Library issues on Solaris with gcc 2.95, so I suggest that at least version 3.2 or above is used.

Part of the reason for using gcc is the ease of interoperability with FORTRAN. The code base makes fairly extensive use of FORTRAN so you need to have g77 installed.
The software is compiled by writing 

<pre>
$ make all
</pre>

at the command line. Architecture specific options are included in the `make.ARCHITECTURE` files. Rename the file with the relevant architecture to `make.inc` for it to be included.

<h3>Optimisation</h3>

One of the advantages of interfacing to the LAPACK and BLAS libraries is that they are often optimised for particular architectures. The file `make.atlas` includes options for compiling the ATLAS optimised versions of lapack and blas that are available on a server I have access to. These options may vary for particular machines.

<h3>Cygwin</h3>

For Windows users the code compiles under cygwin. However you will need version s of the lapack and blas libraries available (see <a href="http://www.netlib.org">www.netlib.org</a>. This can take some time to compile, and in the absence of any pre-compiled versions on the web I've provided some pre-compiled versions you may want to make use of (see the cygwin directory). Note that these pre-compiled versions are <i>not</i> optimised for the specific architecture and therefore do not give the speed up you would hope for from using lapack and blas.

<h3>Microsoft Visual C++</h3>

Thanks to modifications by William V. Baxter the code compiles under Microsoft Visual Studio 7.1. A project file is provided in the current release in the directory `MSVC/ivm`. The compilation makes use of f2c versions of the FORTRAN code and the C version of LAPACK/BLAS, CLAPACK. Detailed instructions on how to compile are in the readme.msvc file. Much of the work to convert the code (which included ironing out several bugs) was done by William V. Baxter for the GPLVM code. 


# IVM

This page describes how to compile and gives some examples of use of the C++ Informative Vector Machine Software (IVM) available for <a href="http://www.cs.man.ac.uk/neill-bin/software/downloadForm.cgi?toolbox=ivmcpp">download here</a>.

<h2>General Information</h2>

The way the software operates is through the command line. There is one executable, `ivm`. Help can be obtained by writing 

<pre>
$ ./ivm -h
</pre>

which lists the commands available under the software. Help for each command can then be obtained by writing, for example, 

<pre>
$ ./ivm learn -h
</pre>

All the tutorial optimisations are suggested take less than 1/2 hour to run on my less than 2GHz Pentium IV machine. The first oil example runs in a couple of minutes. Below I suggest using the highest verbosity options `-v 3` in each of the examples so that you can track the iterations.

<h1>Bugs</h1>

Victor Cheng writes:

<i>" ... I've tested your IVM C++ Gaussian Process tool (IVMCPP0p12 version). It is
quite useful. However, the gnuplot function seems has a problem. Every time
I type the command: "Ivm gnuplot traindata name.model", an error comes out
as: "Unknown noise model!". When I test this function with IVMCPP0p11 IVM,
its fine, but IVMCPP0p11 has another problem that it gives "out of memory"
error in test mode! So I use two vesions simultaneously. "</i> 

I'm working (as of 31/12/2007) on a major rewrite, so it's unlikely that these bugs will be fixed in the near future, however if anyone makes a fix I'll be happy to incorporate it! Please let me know.


<h1>Examples</h1>

The software loads in data in the <a href="http://svmlight.joachims.org/">SVM light</a> format. Anton Schwaighofer has written <a href="http://www.igi.tugraz.at/aschwaig/software.html"> a package</a> which can write from MATLAB to the SVM light format.

<a name="toydata"><h2>Toy Data Sets</h2></a>

In this section we present some simple examples. The results will be visualised using `gnuplot`. It is suggested that you have access to `gnuplot` vs 4.0 or above.

Provided with the software, in the `examples` directory, are some simple two dimensional problems. We will first try classification with these examples.

The first example is data sampled from a Gaussian process with an RBF kernel function with inverse width of 10. The input data is sampled uniformly from the unit square. This data can be learnt with the following command.

<pre>$ ./ivm -v 3 learn -a 200 -k rbf examples/unitsquaregp.svml unitsquaregp.model</pre>

The flag `-v 3` sets the verbosity level to 3 (the highest level) which causes the iterations of the scaled conjugate gradient algorithm to be shown. The flag `-a 200` sets the active set size. The kernel type is selected with the flag `-k rbf`. 

<h3>Gnuplot</h3>

The learned model is saved in a file called `unitsquaregp.model`. This file has a plain text format to make it human readable. Once training is complete, the learned kernel parameters of the model can be displayed using 

<pre>$ ./ivm display unitsquaregp.model

Loading model file.
... done.
IVM Model:
Active Set Size: 200
Kernel Type:
compound kernel:
rbfinverseWidth: 12.1211
rbfvariance: 0.136772
biasvariance: 0.000229177
whitevariance: 0.0784375
Noise Type:
Probit noise:
Bias on process 0: 0.237516
</pre>

Notice the fact that the kernel is composed of an RBF kernel, also known as squared exponential kernel or Gaussian kernel; a bias kernel, which is just a constant, and a white noise kernel, which is a diagonal term. The bias kernel and the white kernel are automatically added to the rbf kernel. Other kernels may also be used, see `ivm learn -h` for details.

For this model the input data is two dimensional, you can therefore visualise the decision boundary using

<pre>$ ./ivm gnuplot examples/unitsquaregp.svml unitsquaregp.model unitsquaregp</pre>

The `unitsquaregp` supplied as the last argument acts as a stub for gnuplot to create names from, so for example (using gnuplot vs 4.0 or above), you can write

<pre>$ gnuplot unitsquaregp_plot.gp</pre>

and obtain the plot shown below
<center><img src="unitsquaregp_plot.png"><br>
The decision boundary learnt for the data sampled from a Gaussian process classification. Note the active points (blue stars) typically lie along the decision boundary.</center><br>

The other files created are `oil100_variance_matrix.dat`, which produces the grayscale map of the log precisions and `oil100_latent_data1-3.dat` which are files containing the latent data positions associated with each label.

<h3>Feature Selection</h3>

Next we consider a simple ARD kernel. The toy data in this case is sampled from three Gaussian distributions. To separate the data only one input dimension is necessary. The command is run as follows,

<pre>$ ./ivm learn -a 100 -k rbf -i 1 examples/ard_gaussian_clusters.svml ard_gaussian_clusters.model</pre>

Displaying the model it is clear that it has selected one of the input dimensions, 

<pre>
Loading model file.
... done.
IVM Model:
Active Set Size: 100
Kernel Type:
compound kernel:
rbfardinverseWidth: 0.12293
rbfardvariance: 2.25369
rbfardinputScale: 5.88538e-08
rbfardinputScale: 0.935148
biasvariance: 9.10663e-07
whitevariance: 2.75252e-08
Noise Type:
Probit noise:
Bias on process 0: 0.745098
</pre>

Once again the results can be displayed as a two dimensional plot,

<pre>$ ./ivm gnuplot examples/ard_gaussian_clusters.svml ard_gaussian_clusters.model ard_gaussian_clusters</pre>

<center><img src="ard_gaussian_clusters_plot.png"><br>
The IVM learnt with an ARD RBF kernel. One of the input directions has been recognised as not relevant.
</center>


<h2>Semi-Supervised Learning</h2>

The software also provides an implementation of the null category noise model described in <a href="http://www.cs.man.ac.uk/neill-bin/publications/bibpage.cgi?keyName=Lawrence:semisuper04">Lawrence and Jordan</a>. 

The toy example given in the paper is reconstructed here. To run it type

<pre>$ ./ivm learn -a 100 -k rbf examples/semisupercrescent.svml semisupercrescent.model
</pre>

The result of learning is

<pre>
Loading model file.
... done.
IVM Model:
Active Set Size: 100
Kernel Type:
compound kernel:
rbfinverseWidth: 0.0716589
rbfvariance: 2.58166
biasvariance: 2.03635e-05
whitevariance: 3.9588e-06
Noise Type:
Ncnm noise:
Bias on process 0: 0.237009
Missing label probability for -ve class: 0.9075
Missing label probability for +ve class: 0.9075
</pre>

and can be visualised using

<pre>$ ./ivm gnuplot examples/semisupercrescent.svml semisupercrescent.model semisupercrescent</pre>

followed by 

<pre>$ gnuplot semisupercrescent_plot.gp</pre>

The result of the visualisation being,

<center><img src="semisupercrescent_plot.png"><img src="semisupercrescent_labels_only_plot.png"><br>The result of semi-supervised learning on the crescent data. At the top is the result from the null category noise model. The bottom shows the result from training only on the labelled data only with the standard probit noise model. Purple squares are unlabelled data, blue stars are the active set. <center>

# Gaussian Process

This page describes how to compile and gives some examples of use of the C++ Gaussian Process code.

<h2>General Information</h2>

The way the software operates is through the command line. There is one executable, `gp`. Help can be obtained by writing 

`$ ./gp -h`

which lists the commands available under the software. Help for each command can then be obtained by writing, for example, 

`$ ./gp learn -h`

All the tutorial optimisations suggested take less than 1/2 hour to run on my less than 2GHz Pentium IV machine. The first oil example runs in a couple of minutes. Below I suggest using the highest verbosity options `-v 3` in each of the examples so that you can track the iterations.

<h1>Examples</h1>

The software loads in data in the <a href="http://svmlight.joachims.org/">SVM light</a> format. This is to provide compatibility with other <a href="/~neill/ivmcpp/">Gaussian Process software</a>. Anton Schwaighofer has written <a href="http://www.igi.tugraz.at/aschwaig/software.html"> a package</a> which can write from MATLAB to the SVM light format.

<a name="spgp1d"><h2>One Dimensional Data Data</h2></a>


Provided with the software, in the `examples` directory, is a one dimensional regression problem. The file is called `spgp1d.svml`. 

First we will learn the data using the following command,

<pre>$ ./gp -v 3 learn -# 100 examples/sinc.svml sinc.model</pre>

The flag `-v 3` sets the verbosity level to 3 (the highest level) which causes the iterations of the scaled conjugate gradient algorithm to be shown. The flag `-# 100` terminates the optimisation after 100 iterations so that you can quickly move on with the rest of the tutorial.

The software will load the data in `sinc.svml`. The labels are included in this file but they are <i>not</i> used in the optimisation of the model. They are for visualisation purposes only.

<h3>Gnuplot</h3>

The learned model is saved in a file called `sinc.model`. This file has a plain text format to make it human readable. Once training is complete, the learned covariance function parameters of the model can be displayed using 


<pre>`$ ./gp display sinc.model</pre>

<pre>
Loading model file.
... done.
Standard GP Model: 
Optimiser: scg
Data Set Size: 40
Kernel Type: 
Scales learnt: 0
X learnt: 0
Bias: 0.106658 

Scale: 1 

Gaussian Noise: 
Bias on process 0: 0
Variance: 1e-06
compound kernel:
rbfinverseWidth: 0.198511
rbfvariance: 0.0751124
biasvariance: 1.6755e-05
whitevariance: 0.00204124
</pre>

Notice the fact that the covariance function is composed of an RBF kernel, also known as squared exponential kernel or Gaussian kernel; a bias kernel, which is just a constant, and a white noise kernel, which is a diagonal term. This is the default setting, it can be changed with flags to other covariance function types, see `./gp learn -h` for details.

For your convenience a `gnuplot` file may generated to visualise the data. First run

<pre>$ ./gp gnuplot -r 400 examples/sinc.svml sinc.model sinc</pre>

The `sinc` supplied as the last argument acts as a stub for gnuplot to create names from, so for example (using gnuplot vs 4.0 or above), you can write

<pre>$ gnuplot sinc_plot.gp</pre>

And obtain the plot shown below

![Sinc Plot](./sinc.png)

Gaussian process applied to sinc data.</center><br>

The other files created are `sinc_error_bar_data.dat`, which produces the error bars and `sinc_line_data.dat` which produces the mean as well as `sinc_scatter_data.dat` which shows the training data.

<h3>Other Data</h3>

You might also want to try a larger data set.

<pre>$ ./gp -v 3 learn -# 100 examples/spgp1d.svml spgp1d.model</pre>

<h3>MATLAB and OCTAVE</h3>

While MATLAB can be horribly slow (and very expensive for non-academic users) it is still a lot easier (for me) to code the visualisation routines by building on MATLAB's graphics facilities. To this end you can load in the results from the MATLAB/OCTAVE GP toolbox for further manipulation. You can download the toolbox from <a href="/~neill/gp">here</a>. Once the relevant toolboxes (you need all the dependent toolboxes) are downloaded you can visualise the results in MATLAB using
<pre>
&gt;&gt; [y, X] = svmlread('sinc.svml')
&gt;&gt; gpReadFromFile('sinc.model', X, y)
&gt;&gt;</pre>

where we have used the <a href="./~neill/svml/">SVML toolbox</a> of Anton Schwaighofer to load in the data.

# GP-LVM

Release Notes

Fixed bug which meant that back constraint wasn't working due to failure to initialise lwork properly for dsysv. 

Fixed bug in gplvm.cpp which meant dynamics wasn't working properly because initialization of dynamics model learning parameter wasn't set to zero.

Thanks to Gustav Henter for pointing out these problems.

<h4>Release 0.2</h4>

In this release we updated the class structure of the gplvm model and
made some changes in the way in which files are stored. This release
is intended as a stopgap before a release version in which fitc, dtc
and variational dtc approximations will be available.

In this release the dynamics model of <a href="http://www.dgp.toronto.edu/~jmwang/gpdm/">Wang <i>et al</i>.</a> has been included. The initial work was done by William V. Baxter, with modifications by me to include the unusual prior Wang suggests in his MSc thesis, scaling of the dynamics likelihood and the ability to set the signal noise ratio. A new example has been introduced for this model below.

As part of the dynamics introduction a MATLAB toolbox for the GP-LVM is available in [this github repository](https://github.com/SheffieldML/GPmat).

Version 0.101 was released 21st October 2005.

This release contained modifications by William V. Baxter to enable the code to work with Visual Studio 7.1.

Version 0.1, was released in late July 2005.

This was the original release of the code.

The way the software operates is through the command line. There is
one executable, `gplvm`. Help can be obtained by writing

<pre>$ ./gplvm -h</pre>

which lists the commands available under the software. Help for
each command can then be obtained by writing, for example,

<pre>$ ./gplvm learn -h</pre>

All the tutorial optimisations suggested take less than 1/2 hour to
run on my less than 2GHz Pentium IV machine. The first oil example
runs in a couple of minutes. Below I suggest using the highest
verbosity options `-v 3` in each of the examples so that
you can track the iterations.

<h1>Examples</h1>

The software loads in data in the <a
href="http://svmlight.joachims.org/">SVM light</a> format. This is to
provide compatibility with other <a href="~neil/ivmapp/">Gaussian
Process software</a>. Anton Schwaighofer has written <a
href="http://www.igi.tugraz.at/aschwaig/software.html"> a package</a>
which can write from MATLAB to the SVM light format.

<a name="oilflow"><h2>Oil Flow Data</h2></a>

In the original NIPS paper the first example was the oil flow data
(see <a href="http://www.ncrg.aston.ac.uk/GTM/3PhaseData.html">this
page</a> for details) sub-sampled to 100 points. I use this data a lot
for checking the algorithm is working so in some senses it is not an
independent `proof' of the model.

Provided with the software, in the `examples` directory,
is a sub-sample of the oil data. The file is called
`oilTrain100.svml`.

First we will learn the data using the following command,

<pre>$ ./gplvm -v 3 learn -# 100 examples/oilTrain100.svml oil100.model</pre>

The flag `-v 3` sets the verbosity level to 3 (the
highest level) which causes the iterations of the scaled conjugate
gradient algorithm to be shown. The flag `-# 100`
terminates the optimisation after 100 iterations so that you can
quickly move on with the rest of the tutorial.

The software will load the data in
`oilTrain100.svml`. The labels are included in this file
but they are <i>not</i> used in the optimisation of the model. They
are for visualisation purposes only.

<h3>Gnuplot</h3>

The learned model is saved in a file called
`oil100.model`. This file has a plain text format to make
it human readable. Once training is complete, the learned kernel
parameters of the model can be displayed using

<pre>$ ./gplvm display oil100.model</pre>

<pre>
Loading model file.
... done.
GPLVM Model:
Data Set Size: 100
Kernel Type:
compound kernel:
rbfinverseWidth: 3.97209
rbfvariance: 0.337566
biasvariance: 0.0393251
whitevariance: 0.00267715
</pre>

Notice the fact that the kernel is composed of an RBF kernel, also
known as squared exponential kernel or Gaussian kernel; a bias kernel,
which is just a constant, and a white noise kernel, which is a
diagonal term. This is the default setting, it can be changed with
flags to other kernel types, see `gplvm learn -h` for
details.

For your convenience a `gnuplot` file may generated to
visualise the data. First run

<pre>$ ./gplvm gnuplot oil100.model oil100</pre>

The `oil100` supplied as the last argument acts as a
stub for gnuplot to create names from, so for example (using gnuplot
vs 4.0 or above), you can write

<pre>$ gnuplot oil100_plot.gp</Pre>

And obtain the plot shown below
![Oil 100 Image](./oil100_plot.png)
Visualisation of 100 points of the oil flow data.</center><br>

The other files created are
`oil100_variance_matrix.dat`, which produces the grayscale
map of the log precisions and `oil100_latent_data1-3.dat`
which are files containing the latent data positions associated with
each label.

### The Entire Oil Data Set

Running the GPLVM for 1000 iterations on all 1000 points of the oil
data leads to the visualisation below.

![Full oil data](./oil1000_plot.png)

All 1000 points of the oil data projected into latent space. This visualisation takes overnight to optimise on a Pentinum IV.

### MATLAB

While MATLAB can be horribly slow (and very expensive for non-academic
users) it is still a lot easier (for me) to code the visualisation
routines by building on MATLAB's graphics facilities. To this end a
new release of the GPLVM code in MATLAB has been provided (vs 2.012
and above) which allows you to load the results of the learning from
the C++ code into MATLAB for further manipulation. You can download
the toolbox from <a
href="http://www.cs.man.ac.uk/neill-bin/software/downloadForm.cgi?toolbox=gplvm">here</a>. Once
the relevant toolboxes (you need the IVM toolbox and the toolboxes on
which it depends: KERN, NOISE, etc.) are downloaded you can visualise
the results in MATLAB using

<pre>&gt;&gt; gplvmResultsCpp('oil100.model', 'vector')
&gt;&gt;</pre>

This will load the results and allow you to move around the latent
space visualising (in the form of a line plotted from the vector) the
nature of the data at each point.

<a name="mocap"><h2>Motion Capture</h2></a>

One popular use of the GPLVM has been in learning of human motion
styles (see <a
href="http://www.cs.man.ac.uk/neill-bin/publications/bibpage.cgi?keyName=Grochow:styleik04&printAbstract=1">Grochow
<i>et al.</i></a>). Personally, I find this application very
motivating as Motion Capture data is a rare example of high
dimensional data about which humans have a strong intuition. If the
model fails to model `natural motion' it is quite apparant to a human
observer. Therefore, as a second example, we will look at data of this
type. In particular we will consider a data sets containing a walking
man and a further data set containing a horse. To run these demos you
will also need a small <a
href="http://www.cs.man.ac.uk/neill-bin/software/downloadForm.cgi?toolbox=mocap">MATLAB
mocap toolkit</a>.

<a name="mocap"><h3>BVH Files</h3></a>

To prepare a new bvh file for visualisation you need the MATLAB
mocap toolkit and Anton Schwaighofer's <a
href="http://www.igi.tugraz.at/aschwaig/software.html"> SVM light
MATLAB interface</a> (you don't need the SVM light software itself).

<pre>&gt;&gt; [bvhStruct, channels, frameLength] = bvhReadFile('examples/Swagger.bvh');
&gt;&gt;</pre>

This motion capture data was taken from Ohio State University's <a
href="http://accad.osu.edu/research/mocap/mocap_data.htm">ACCAD</a>
centre.

The motion capture channels contain values for the offset of the
root node at each frame. If we don't want to model this motion it can
be removed at this stage. Setting the 1st, 3rd and 6th channels to
zero removes X and Z position and the rotation in the Y plane.

<pre>&gt;&gt; channels(:, [1 3 6]) = zeros(size(channels, 1), 3);</pre>

You can now play the data using the command

<pre>&gt;&gt; bvhPlayData(bvhStruct, channels, frameLength);</pre>

Data in the bvh format consists of angles, this presents a problem
when the angle passes through a discontinuity. For example in this
data the 'lhumerus' and 'rhumerus' joints rotate through 180 degrees
and the channel moves from -180 to +180. This arbitrary difference
will seriously effect the results. The fix is to add or subtract 360
as appropriate. In the toolbox provided this is done automatically in
the file bvhReadFile.m using the function channelsAngles. This works
well for the files we use here, but may not be a sufficient solution
for files with more rotation on the joints.

Then channels can be saved for modelling using Schwaighofer's SVM
light interface. First we downsample so that things run quickly,

<pre>&gt;&gt; channels = channels(1:4:end, :);</pre>

Then the data is saved as follows:

<pre>&gt;&gt; svmlwrite('examples/swagger.svml',channels)</pre>

<i>Before you save you might want to check you haven't messed
anything up by playing the data again!</i> It makes sense to learn
the scale independently for each the channels (particularly since we
have set three of them to zero!), so we now use the gplvm code to
learn the data setting the flag `-L true` for learning of
scales.

<pre>$ ./gplvm -v 3 learn -L true examples/swagger.svml swagger.model</pre>

Once learning is complete the results can be visualised in MATLAB
using the command

<pre>&gt;&gt; mocapResultsCppBvh('swagger.model', 'examples/Swagger.bvh', 'bvh');</pre>

<center><img src="swagger_plot.png">
<br>Latent space for the Swagger data. Note the breaks in the sequence.</center>

<h3>Dealing with the Breaks</h3>

Note that there are breaks in the sequence. These reason for these
breaks is as follows. The GPLVM maps from the latent space to the data
space with a smooth mapping. This means that points that are nearby in
latent space will be nearby in data space. However that does not imply
the reverse, i.e. points that are nearby in data space will not
necessarily be nearby in latent space. It implies that if points are
far apart in data space they will be far apart in latent space which
is a slightly different thing. This means that the model is not
strongly penalised for breaking the sequence (even if a better
solution can be found through not breaking the sequence).

For visualisation you often want points being nearby in data space to
be nearby in latent space. For example, most of the recent spectral
techniques (including kernel PCA, Isomap, and LLE) try and guarantee
this. In recent unpublished work with <a
href="http://www.kyb.tuebingen.mpg.de/~jqc">Joaquin Quinonero
Candela</a>, we have shown that this can be achieved in the GPLVM
using `back constraints'. We constrain the data in the latent space to
be represented by a second reverse-mapping from the data space. For
the walking man the show results you can test the back constraints
with the command


<pre>$ ./gplvm -v 3 learn -L true -c rbf -g 0.0001 examples/swagger.svml swagger_back_constrained.model</pre>

The back constraint here is a kernel mapping with an `RBF' kernel
which is specified as having an inverse width of 1e-4.
 
The results can then be seen in MATLAB using

<pre>&gt;&gt; mocapResultsCppBvh('swagger_back_constrained.model', 'examples/Swagger.bvh', 'bvh');</pre>

<center><img src="swagger_back_constrained_plot.png">
<br>The repeated circular pattern is associated with the repeated walking paces in the data.</center>

<h3>Dealing with the Breaks with Dynamics</h3>

It conceptually straightforward to add MAP dynamics in the GP-LVM
latent space by placing a prior that relates the latent points
temporally. There are several ways one could envisage doing this. <a
href="http://www.dgp.toronto.edu/~jmwang/gpdm/">Wang <i>et al</i></a>
proposed introducing dynamics through the use of a Gaussian process
mapping across time points. William V. Baxter implemented this
modification to the code and kindly allowed me to make his
modifications available. In the base case adding dynamics associated
with a GP doesn't change things very much: the Gaussian process
mapping is too flexible and doesn't constrain the behaviour of the
model. To solve this problem Wang suggests using a particular prior on
the hyper parameters of the GP-LVM (see pg 58 of his <a
href="http://www.dgp.toronto.edu/~jmwang/gpdmthesis.pdf">Master's
thesis</a>). This prior is unusual as it is improper, but it is not
the standard uninformative 1/x prior. This approach can be recreated
using the `-dh` flag when running the code. An alternative
approach of scaling the portion of the likelihood associated with the
dynamics up by a factor has also been suggested. This approach can be
recreated by using the `-ds` flag.

My own preference is to avoid either of these approaches. A key
motivation of the GP-LVM as a probabilistic model was to design an
approach that avoided difficult to justify scalings and unusual
priors. The basic problem is that if the hyper parameters are
optimised the Gaussian process is too flexible a model for application
modelling the dynamics. However it is also true that a non-linear
model is needed. As an alternative approach we suggest fixing the
hyper parameters. The level of noise can be fixed by suggesting a
signal to noise ratio. This approach has also been implemented in the
code using the `-dr` flag.

<pre>$ ./gplvm -v 3 learn -L true -D rbf -g 0.01 -dr 20 examples/swagger.svml swagger_dynamics.model</pre>

where the `-M` flag sets the parameter associated with
Wang's prior. Here the dynamics GP is given a linear and an RBF
kernel. The results of the visualisation are shown below.

<center><img src="swagger_dynamics_plot.png"> <br>Latent space for
the Swagger data with the dynamics. By constraining the GP-LVM with an
unusual prior the sequence stays continuous in latent space.</center>

This result can also be loaded into MATLAB and played using the command 

<pre>&gt;&gt; mocapResultsCppBvh('swagger_dynamics.model', 'examples/Swagger.bvh', 'bvh');</pre>

