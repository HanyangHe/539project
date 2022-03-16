clear
clc

global person0
global person1
global run
global SmallCelebrityImageData
global aim
global PCA_ap
global PCA_patK
global CARC_ap
global CARC_patK

% In this code, the author provide a new feature process link after PCA to
% improve the identification performance
fprintf('Loading dataset...\n');
load('SmallCelebrityData.mat');
load('SmallCelebrityImageData.mat');


eps = 10^-5;
maxrun=1;% maxrun=1 to cancel this function

aim=zeros(1,maxrun);
aim(1)=0;

% nPart = zeros(1,maxrun);% The part number of PCA (default=16)
% pcaDim = zeros(1,maxrun);% The dimension of PCA (default=500)
% featureNumOfPerImage=size(SmallCelebrityImageData.feature,2);% the number of features for one image
% partDim = featureNumOfPerImage/nPart;% The number of per image feature in each PCA part
% pcaDim<=partDim
pcaDimMax=size(SmallCelebrityImageData.feature,1);
nPartMax=featureNumOfPerImage/pcaDimMax;
% nPart=1:featureNumOfPerImage/pcaDimMax;
% pcaDim=1:1:pcaDimMax;

step=zeros(1,maxrun);
theta=zeros(1,maxrun);
step(1)=5;
theta(1)=0;%rad

person0=1;%left boundary of person index used in the experiment
person1=10;%right boundary of person index used in the experiment

%=============================calculation==================================
for run=1:maxrun
    disp(['Turn: ',num2str(run)])
    % x = ga(@fitnessFCN(nPart,pcaDim),2);%matlab original ga can only used to
    % find minimum value's independent variable set in a given solution space
    GAVarRange=[1/nPartMax nPartMax/nPartMax;1/pcaDimMax pcaDimMax/pcaDimMax];%first row is the range of nPart
                                                        %second row is the
                                                        %range of pcaDim
    fprintf('initializega')
    initPop=initializega(10,GAVarRange,'fitnessFCN');
    fprintf('ga')
    [x,endPop,bpop,trace]=ga(GAVarRange,'fitnessFCN',[],initPop,[1e-3 1 0],'maxGenTerm',25 ...
        ,'normGeomSelect',0.08,'arithXover',2,'nonUnifMutation',[2 25 3]);
    %The GA algorithm is suitable for invisible complex model functions, 
    % with good globality, but time-consuming and unstable performance.
end

best=find(aim==max(aim));
disp(['The best nPart is: ',num2str(floor(x(1)*nPartMax))])
disp(['The best pcaDim is: ',num2str(floor(x(2)*pcaDimMax))])
for i=1:3
    fprintf('High-Dimensional LBP:\tMAP = %f, P@1 = %f\n', PCA_ap{best(1),i}, PCA_patK{best(1),i});
    fprintf('CARC:\t\t\tMAP = %f, P@1 = %f\n', CARC_ap{best(1),i}, CARC_patK{best(1),i});
end

fprintf('High-Dimensional LBP:\tMAP_ave = %f, P@1_ave = %f\n', (PCA_ap{best(1),1}+PCA_ap{best(1),2}+PCA_ap{best(1),3})/3, (PCA_patK{best(1),1}+PCA_patK{best(1),2}+PCA_patK{best(1),3})/3);
fprintf('CARC:\t\t\tMAP_ave = %f, P@1_ave = %f\n', (CARC_ap{best(1),1}+CARC_ap{best(1),2}+CARC_ap{best(1),3})/3, (CARC_patK{best(1),1}+CARC_patK{best(1),2}+CARC_patK{best(1),3})/3);