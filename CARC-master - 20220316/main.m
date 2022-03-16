clear
clc
global nPart
global pcaDim
global person0
global person1
global run
% In this code, the author provide a new feature process link after PCA to
% improve the identification performance
fprintf('Loading dataset...\n');
load('SmallCelebrityData.mat');
load('SmallCelebrityImageData.mat');


eps = 10^-5;
Derror=0.000001;%optimal solve error
maxrun=1;

aim=zeros(1,maxrun);
aim(1)=0;

nPart = zeros(1,maxrun);% The part number of PCA (default=16)
pcaDim = zeros(1,maxrun);% The dimension of PCA (default=500)
nPart(1)=16;
pcaDim(1)=500;

step=zeros(1,maxrun);
theta=zeros(1,maxrun);
step(1)=5;
theta(1)=0;%rad

% rng(1);

for run=1:maxrun

person0=1;%left boundary of person index used in the experiment
person1=10;%right boundary of person index used in the experiment

featureNumOfPerImage=size(SmallCelebrityImageData.feature,2);% the number of features for one image
partDim = floor(featureNumOfPerImage/nPart(run));% The number of per image feature in each PCA part
cPts = size(SmallCelebrityImageData.identity,1);%The total number of the pictures
SmallCelebrityImageData.pcaFeature = zeros(cPts, pcaDim(run)*nPart(run));% Initial the size of the PCA feature matrix
changeIndex = reshape([1:featureNumOfPerImage], [], 5)';
changeIndex = changeIndex(:);% this is a special way to shuffle the index

%=============pca======================
fprintf('Computing PCA feature...\n');
% The method divide the whole features of a image in to n part
for p = 1:nPart(run)% execute PCA in each small part one by one
   partIndex = changeIndex([1 + (p-1)*partDim:p*partDim]);% Index sets of feature in each part for pca
   pcaIndex = [1 + (p-1)*pcaDim(run):p*pcaDim(run)];% Index sets of feature after the pca process

   % PCA (only person's index number >k use PCA) without normalization
%    X = double(SmallCelebrityImageData.feature(:,partIndex));
%    [score_mappedX, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim);
%    X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;

   % PCA (only person's index number >k use PCA) with normalization
   X = double(SmallCelebrityImageData.feature(:,partIndex));
   X=(X-ones(size(X,1),1)*mean(X))./(ones(size(X,1),1)*std(X));% N(0,1) normalization of each image feature
   [score_mappedX, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim(run));
   X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;

   %original version PCA (strange data process step)
%    X = sqrt(double(SmallCelebrityImageData.feature(:,partIndex)));
%    [~, PCAmapping] = pca(X(find(SmallCelebrityImageData.rank > 0), :), pcaDim);
%    X_PCA = bsxfun(@minus, X, PCAmapping.mean) * PCAmapping.M;
%    W = diag(ones(pcaDim,1)./sqrt(PCAmapping.lambda + eps));
%    X_PCA = X_PCA*W;

   SmallCelebrityImageData.pcaFeature(:,pcaIndex) = X_PCA;%save all part pca result in sequence into pcaFeature
end

%====================LDA=========================
% ldaDim=pcaDim;
% for p = 1:nPart% execute LDA in each small part one by one
%    ldaIndex = [1 + (p-1)*ldaDim:p*ldaDim];
%    X = sqrt(double(SmallCelebrityImageData.feature(:,partIndex)));% sqrt
% 
%    [score_mappedX, PCAmapping] = myLDA(X(find(SmallCelebrityImageData.rank > 0), :), ldaDim);
%    X_PCA = ;
% 
%    SmallCelebrityImageData.pcaFeature(:,ldaIndex) = X_PCA;%save all part pca result in sequence into pcaFeature
% end


fprintf('Computing Cross-Age Reference Coding...\n');
%choose 2004-2012 3 year-layer-devided people who's indexes are person0~person1 as train set
databaseIndex{1} = find((SmallCelebrityImageData.year == 2004 | SmallCelebrityImageData.year == 2005 | SmallCelebrityImageData.year == 2006) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex{2} = find((SmallCelebrityImageData.year == 2007 | SmallCelebrityImageData.year == 2008 | SmallCelebrityImageData.year == 2009) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
databaseIndex{3} = find((SmallCelebrityImageData.year == 2010 | SmallCelebrityImageData.year == 2011 | SmallCelebrityImageData.year == 2012) & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
%choose 2013 people who's index are person0~person1 as a test set
queryIndex = find(SmallCelebrityImageData.year == 2013 & SmallCelebrityImageData.rank <=person1 & SmallCelebrityImageData.rank >= person0);
lambda = 10;
lambda2 = 10000;
CARC_query = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, queryIndex);
CARC_database{1} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex{1});
CARC_database{2} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex{2});
CARC_database{3} = CARC(SmallCelebrityImageData, SmallCelebrityData, lambda, lambda2, databaseIndex{3});
dataset{1} = '2004-2006';
dataset{2} = '2007-2009';
dataset{3} = '2010-2012';

%Here is for prepareing your own features, the order of the features should be same as "image.list"
%{
SmallcelebrityImageData.newFeature = zeros(featureNumOfPerImage, feature_dim);
%}

fprintf('Evaluation...\n');
queryId = SmallCelebrityImageData.identity(queryIndex);
for i = 1:3
%    fprintf(['Result for dataset ' dataset{i} '\n']);
   databaseId = SmallCelebrityImageData.identity(databaseIndex{i});
   
   % High-Dimensional LBP method
   qX = SmallCelebrityImageData.pcaFeature(queryIndex, :);
   X = SmallCelebrityImageData.pcaFeature(databaseIndex{i}, :);
   dist = -1*normalizeL2(qX)*normalizeL2(X)';%normalizeL2: Eular distance standarlization of X
   %'dist' reflect a distance index of a test image to each image used as reference 
   result = evaluation(dist, queryId, databaseId);%mean average precision
%    fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   PCA_ap{run,i}=mean(result.ap);
   PCA_patK{run,i}=result.patK(1);
   
   % LDA method
%    qX2 = SmallcelebrityImageData.ldaFeature(queryIndex, :);
%    X2 = SmallcelebrityImageData.ldaFeature(databaseIndex{i}, :);
%    dist = -1*normalizeL2(qX2)*normalizeL2(X2)';
%    result = evaluation(dist, queryId, databaseId);
%    fprintf('LDA Features:\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));

   % CARC method
   dist = -1*normalizeL2(CARC_query)*normalizeL2(CARC_database{i})';
   result = evaluation(dist, queryId, databaseId);
%    fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', mean(result.ap), result.patK(1));
   aim(run)=aim(run)+mean(result.ap);
   CARC_ap{run,i}=mean(result.ap);
   CARC_patK{run,i}=result.patK(1);
end
if run==1
    nPart(run+1)=nPart(run)+sin(theta(run))*step(run);
    pcaDim(run+1)=pcaDim(run)+cos(theta(run))*step(run);
else
%     if aim(run)>=aim(run-1)%right direction (confidence: small theta change, bigger step)
%         theta(run)=theta(run-1)+(rand(1)-0.5)*2*pi/8;%random choose +-pi/4 angle
%         step(run)=step(run-1)*1.25;
%         nPart(run+1)=nPart(run)+sin(theta(run))*step(run);
%         pcaDim(run+1)=pcaDim(run)+cos(theta(run))*step(run);
%     else%wrong direction (cautious: reverse, smaller step)
%         theta(run)=theta(run-1)+(rand(1)-0.5)*2*(pi/8+pi);%random choose pi+-pi/4 angle
%         step(run)=step(run-1)*0.75;
%         nPart(run+1)=nPart(run)+sin(theta(run))*step(run);
%         pcaDim(run+1)=pcaDim(run)+cos(theta(run))*step(run);
%     end
%     if aim(run)-aim(run-1)<Derror && run>7
%         break
%     end
    
end
nPart(run+1)=round(nPart(run+1));
pcaDim(run+1)=round(pcaDim(run+1));
if pcaDim(run+1)>size(SmallCelebrityImageData.feature,1)
    pcaDim(run+1)=size(SmallCelebrityImageData.feature,1);
end
end
best=find(aim==max(aim));
disp(['The best nPart is: ',num2str(nPart(best(1)))])
disp(['The best pcaDim is: ',num2str(pcaDim(best(1)))])
for i=1:3
    fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', PCA_ap{run,i}, PCA_patK{run,i});
    fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', CARC_ap{run,i}, CARC_patK{run,i});
end