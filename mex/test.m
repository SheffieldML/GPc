% test.m tests the functionality if the compiled fGP mex-file and shows basic work with it
% 
% svml tools need to be accessible on the path

%% load the training data
[yt, xt] = svmlread('../examples/spgp1d.svml');

%% train, query and clear the model

fprintf('Training the model...');
tic;
fGP('train', 'rBw', xt(1:300), yt(1:300), 0);
t = toc;
fprintf('done in %.0f ms.\n', t*1000);

fprintf('Retraining the model...');
tic;
fGP('retrain', xt, yt, 0);
t = toc;
fprintf('done in %.0f ms.\n', t*1000);

xq = (-1.5:0.01:1.5)';
fprintf('Querying the model...');
tic;
[yq, yVar] = fGP('query', xq);
t = toc;
fprintf('done in %.0f ms.\n', t*1000);

fprintf('Clearing the model...');
tic;
fGP('clear');
t = toc;
fprintf('done in %.0f ms.\n', t*1000);

%% plot the prediction (including training data)

stdM = 1; % standard deviation multiplier (size of error bars)

figure(1);
set(gcf, 'Name', 'GPc model with RBF+bias+whiteNoise kernel');
hold off

% confidence interval
yStd = sqrt(yVar);
lowerBound = yq - stdM * yStd;
upperBound = yq + stdM * yStd;
ah = area(xq, [lowerBound, upperBound-lowerBound], 'EdgeColor', 'none');
set(ah(1), 'FaceColor', 'none');
set(ah(2), 'FaceColor', [.7 .8 1]);
% set(get(ah(2), 'Children'), 'FaceAlpha', 0.5);
hold on

% training data
th = scatter(xt, yt, 'g+');

% mean prediction
mh = plot(xq, yq, 'b', 'LineWidth', 2);

legend([th, mh, ah(2)], {'Training data', 'Mean prediction', 'Prediction conf. int.'},...
	'Location', 'NE');

%% print the plot

set(gcf, 'PaperUnits', 'centimeters', 'PaperPosition', [0, 0, 16.18, 10]);

print -dpng spgp1d_mex.png

