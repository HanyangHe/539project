function dist=EularDistOfFeatureMat(Test,Train)
% calculate the Eular distance between two matrix have the same feature (column number)
% Test is test matrix (row is sample, column is feature)
% Train is train matrix (row is sample, column is feature)
% dist is the distance matrix between Test and Train (a row is a Test sample, a column is a Train sample)

TestSpNum=size(Test,1);
TrainSpNum=size(Train,1);
% featureNum=size(Test,2);% = size(Train,2)
dist=zeros(TestSpNum,TrainSpNum);

for i=1:TestSpNum
    Test_i=repmat(Test(i,:),TrainSpNum,1);
    dist(i,:)=sqrt(sum((Test_i-Train).^2,2))';
end
dist;