function [TrainDR,TestDR,Cls] = klda (TRData,TEData)
%-----------------------------------------------------------------------
% Kernel LDA for Dimensionality Reduction
%-----------------------------------------------------------------------
% Performs Dimensionality Reduction for the given Dataset
% using Kernel Linear Discriminant Analysis technique
%-----------------------------------------------------------------------
% Inputs = (TRData,TEData)
% TRData is the Matrix containing Training Data of size P*(M+1)
% TEData is the Matrix containing Test Data Q*M
% where P = Number of Instances in Training Data
% where Q = Number of Instances in Test Data
% where M = Number of Attributes (Features) of Dataset
% Rows are Instances and Columns are Features of Dataset
% Last column of Training Data denotes the Class of the instance
%-----------------------------------------------------------------------
% Output = [TrainDR,TestDR,Cls]
% TrainDR is the dimensionally reduced form of Training Data
% TestDR is the dimensionally reduced form of Test Data
% Rows are Instances of Dataset, and remains the same
% Columns are Features of Dataset, which is reduced
% Cls is the column vector indicating class of each training vector
%-----------------------------------------------------------------------

%% Data formatting
Cls = TRData(:,end);  % Class feature extraction
Ucls = unique(Cls);  % Extraction of distinct classes
TRData = TRData(:,1:1:end-1);  % Removing class feature for training


%% Gram matrix
P = size(TRData,1);  % Training Data Instances
sg = 0.005;  % Standard Deviation of Gaussian Kernel
KGM = zeros(P);  % Gaussian Kernel Gram Matrix
for i = 1:1:P
    for j = 1:1:P
        t1 = TRData(i,:);
        t2 = TRData(j,:);
        t1 = t1/sqrt(sum(t1.^2));
        t2 = t2/sqrt(sum(t2.^2));
        KGM(i,j) = exp( -sum((t1-t2).^2) / (sg^2));
    end
end


%% Test matrix
Q = size(TEData,1);  % Test Data Instances
TM = zeros(Q,P);  % Test Matrix
for i = 1:1:Q
    for j = 1:1:P
        t1 = TEData(i,:);
        t2 = TRData(j,:);
        t1 = t1/sqrt(sum(t1.^2));
        t2 = t2/sqrt(sum(t2.^2));
        TM(i,j) = exp( -sum((t1-t2).^2) / (sg^2));
    end
end


%% Class separation
DCell = cell(length(Ucls),1);  % Grouping class vectors into cells
for p = 1:1:length(Ucls)
    DCell{p} = KGM(Cls==Ucls(p),:);
end


%% Scatter matrix
Sw = 0;  % Within-Class Scatter Matrix
Mn = zeros(length(Ucls),P);
for p = 1:1:length(Ucls)
    Sw = Sw + cov(DCell{p});
    Mn(p,:) = mean(DCell{p});
end
Sb = cov(Mn);  % Between-Class Scatter Matrix


%% Dimensionality reduction
[V,E] = eig(pinv(Sw)*Sb);  % Eigenvalues and Eigenvectors
E = E/max(max(E));
[E,I] = sort(diag(E),'descend');
thr = 0.001;  % Normalised Threshold for choosing eigenvalue
Te = E(E>thr);
W = V(:, I(1:1:length(Te)) );  % Discriminant Matrix
TrainDR = KGM * W;  % Dimensionally Reduced Training Data
TestDR = TM * W;  % Dimensionally Reduced Test Data


%% Feature Scaling
Ni = size(TrainDR,2);
MTD = zeros(1,Ni);
SDVTD = zeros(1,Ni);
for j = 1:1:Ni
    MTD(j) = mean(TrainDR(:,j));
    SDVTD(j) = 0.5*std(TrainDR(:,j));
    TrainDR(:,j) = (TrainDR(:,j) - MTD(j))/SDVTD(j);
end
Nt = size(TestDR,1);
TestDR = (TestDR - repmat(MTD,[Nt,1]))./repmat(SDVTD,[Nt,1]);


%% 3D Plot of Training Data
r = TrainDR;
N = size(TrainDR,2);  % Lower Dimension

if(N==2)
    r(:,3) = zeros(size(r,1),1);
else
    if(N==1)
        r(:,[2,3]) = zeros(size(r,1),2);
    end
end

% 3D Scatter Plot
Syc={'o','x','+','s'};
t=0;
lgdtxt = cell(length(Ucls),1);
figure;
hold on;
for p = 1:1:length(Ucls)
    tn = t + sum(Cls==Ucls(p));
    lgdtxt{p} = ['Class ', num2str(p), ' vectors'];
    scatter3(r(t+1:1:tn,1),r(t+1:1:tn,2),r(t+1:1:tn,3),Syc{p});
    t = tn;
end
title('3D Plot of Dimensionally Reduced Vectors from Training Data');
xlabel('x axis');
ylabel('y axis');
zlabel('z axis');
legend(lgdtxt);
view(30,30);

end