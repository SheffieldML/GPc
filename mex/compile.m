% compile.m compiles given source to a Matlab-executable mexfile, using the GPc library.
% 
% In case Matlab complains about the dynamically linked library libgp not
% being found, try either to force link to the static one or setting
% LD_LIBRARY_PATH environment variable.

mexName = 'fGP';
cppExt = '.mex.cpp';
gpDir = '..';

mex([mexName cppExt],...
	'-output', [mexName '.' mexext],...
	['-I' gpDir],...
	'-lgp', ['-L' gpDir],...
	'-lblas', '-llapack', '-lgfortran');


