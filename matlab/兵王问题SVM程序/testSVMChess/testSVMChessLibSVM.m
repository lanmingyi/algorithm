clear all;
% Read the data. 读取、转换数据
fid  =  fopen('krkopt.DATA');
c = fread(fid, 3);

vec = zeros(6,1);
xapp = [];  % 接收6维的训练数据
yapp = [];  % 接收对应的标签
while ~feof(fid)
    string = [];
    c = fread(fid,1);
    flag = flag+1;
    while c~=13
        string = [string, c];
        c=fread(fid,1);
    end;
    fread(fid,1);  
    if length(string)>10
        vec(1) = string(1) - 96;
        vec(2) = string(3) - 48;
        vec(3) = string(5) - 96;
        vec(4) = string(7) - 48;
        vec(5) = string(9) - 96;
        vec(6) = string(11) - 48;
        xapp = [xapp,vec];
        if string(13) == 100
            yapp = [yapp,1];
        else
            yapp = [yapp,-1];
        end;
    end;
end;
fclose(fid);

[N,M] = size(xapp);
p = randperm(M);  %直接打乱了训练样本，获得随机的整数排列
numberOfSamplesForTraining = 5000;
xTraining = [];
yTraining = [];
for i=1:numberOfSamplesForTraining
    xTraining  = [xTraining,xapp(:,p(i))];  % 训练集
    yTraining = [yTraining,yapp(p(i))];
end;
xTraining = xTraining';
yTraining = yTraining';

xTesting = [];
yTesting = [];
for i=numberOfSamplesForTraining+1:M
    xTesting  = [xTesting,xapp(:,p(i))];  % 测试集
    yTesting = [yTesting,yapp(p(i))];
end;
xTesting = xTesting';
yTesting = yTesting';

%%%%%%%%%%%%%%%%%%%%%%%%
%Normalization
[numVec,numDim] = size(xTraining);
avgX = mean(xTraining);
stdX = std(xTraining);
for i = 1:numVec
    xTraining(i,:) = (xTraining(i,:)-avgX)./stdX;  % 归一化操作，减去均值再除以方差
end;
[numVec,numDim] = size(xTesting);

for i = 1:numVec
    xTesting(i,:) = (xTesting(i,:)-avgX)./stdX;
end;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%SVM Gaussian kernel （RBF核）
%Search for the optimal C and gamma, K(x1,x2) = exp{-||x1-x2||^2/gamma} to
%make the recognition rate maximum. 

%Firstly, search C and gamma in a crude scale (as recommended in 'A practical Guide to Support Vector Classification'))
CScale = [-5, -3, -1, 1, 3, 5,7,9,11,13,15];
gammaScale = [-15,-13,-11,-9,-7,-5,-3,-1,1,3];
C = 2.^CScale;
gamma = 2.^gammaScale;
maxRecognitionRate = 0;  % 82-93：训练集数据进行交叉验证，找出识别率最高的C和gamma组合
for i = 1:length(C)
    for j = 1:length(gamma)
        cmd=['-t 2 -c ',num2str(C(i)),' -g ',num2str(gamma(j)),' -v 5'];  % -v 5：进行5折交叉验证
        recognitionRate = svmtrain(yTraining,xTraining,cmd);
        if recognitionRate>maxRecognitionRate
            maxRecognitionRate = recognitionRate
            maxCIndex = i;
            maxGammaIndex = j;
        end;
    end;
end;

%Then search for optimal C and gamma in a refined scale. 
n = 10;  % 96-117：进一步缩小搜索范围，再次使用交叉验证找识别率更高更精确的C和gamma组合
minCScale = 0.5*(CScale(max(1,maxCIndex-1))+CScale(maxCIndex));
maxCScale = 0.5*(CScale(min(length(CScale),maxCIndex+1))+CScale(maxCIndex));
newCScale = [minCScale:(maxCScale-minCScale)/n:maxCScale];

minGammaScale = 0.5*(gammaScale(max(1,maxGammaIndex-1))+gammaScale(maxGammaIndex));
maxGammaScale = 0.5*(gammaScale(min(length(gammaScale),maxGammaIndex+1))+gammaScale(maxGammaIndex));
newGammaScale = [minGammaScale:(maxGammaScale-minGammaScale)/n:maxGammaScale];
newC = 2.^newCScale;
newGamma = 2.^newGammaScale;
maxRecognitionRate = 0;
for i = 1:length(newC)
    for j = 1:length(newGamma)
        cmd=['-t 2 -c ',num2str(newC(i)),' -g ',num2str(newGamma(j)),' -v 5'];
        recognitionRate = svmtrain(yTraining,xTraining,cmd);
        if recognitionRate>maxRecognitionRate
            maxRecognitionRate = recognitionRate
            maxC = newC(i);
            maxGamma = newGamma(j);
        end;
    end;
end;

%Train the SVM model by the optimal C and gamma. C和gamma确定后，利用训练集所有样本重新训练一个支持向量机模型（最终模型）
cmd=['-t 2 -c ',num2str(maxC),' -g ',num2str(maxGamma)];
model = svmtrain(yTraining,xTraining,cmd);
save model.mat model;

%Test the model on the remaining testing data and obtain the recognition rate.
load model.mat;
[yPred,accuracy,decisionValues] = svmpredict(yTesting,xTesting,model); 
save yPred.mat yPred;
save decisionValues.mat decisionValues;
save xTraining.mat xTraining;
save yTesting.mat yTesting;

%%draw ROC
[totalScores,index]  = sort(decisionValues);
labels = yTesting;
for i = 1:length(labels)
    labels(i) = yTesting(index(i));
end;


truePositive = zeros(1,length(totalScores)+1);
trueNegative = zeros(1,length(totalScores)+1);
falsePositive = zeros(1,length(totalScores)+1);
falseNegative = zeros(1,length(totalScores)+1);

for i = 1:length(totalScores)
    if labels(i) == 1
        truePositive(1) = truePositive(1)+1;
    else
        falsePositive(1) = falsePositive(1) +1;
    end;
end;

for i = 1:length(totalScores)
   if labels(i) == 1
       truePositive(i+1) = truePositive(i)-1;
       falsePositive(i+1) = falsePositive(i);
   else
       falsePositive(i+1) = falsePositive(i)-1;
       truePositive(i+1) = truePositive(i);
   end;
end;
truePositive = truePositive/truePositive(1);
falsePositive = falsePositive/falsePositive(1);
plot(falsePositive,truePositive);



