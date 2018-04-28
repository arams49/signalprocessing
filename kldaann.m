%% Dimensionality Reduction and Classification using KLDA + ANN
%% Function to perform KLDA and ANN
%-----------------------------------------------------------------------
% Dimensionality Reduction using Kernel Linear Discriminant Analysis
% Classification using Artificial Neural Network
% Written by Abhiram S
%-----------------------------------------------------------------------

%% Dataset Load
filename = 'Meter C';
Data = load(filename);  % Load the appropriate Dataset
K = floor(0.15*size(Data,1));  % No. of instances in Test set
tek = randperm(size(Data,1),K);  % Random instances for Test set
TEData = Data(tek,:);  % Test Data Set
Tecl = TEData(:,end);  % Classes of Test Data Set
TEData = TEData(:,1:1:end-1);  % Removing class feature for testing
TRData = Data;  % Training Data Set


%% Kernel LDA Function
%-----------------------------------------------------------------------
% Performs Dimensionality Reduction for the given Dataset
% using Kernel Linear Discriminant Analysis technique
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


%% Feature Scaling and Normalization
Ni = size(TrainDR,2);  % Number of reduced features
MTD = zeros(1,Ni);  % Mean of attributes
SDVTD = zeros(1,Ni);  % Standard Deviation of attributes
for j = 1:1:Ni
    MTD(j) = mean(TrainDR(:,j));
    SDVTD(j) = 0.5*std(TrainDR(:,j));
    TrainDR(:,j) = (TrainDR(:,j) - MTD(j))/SDVTD(j);
end
Nt = size(TestDR,1);
TestDR = (TestDR - repmat(MTD,[Nt,1]))./repmat(SDVTD,[Nt,1]);


%% 3D Plot of Training Data
r = TrainDR;

if(Ni==2)
    r(:,3) = zeros(size(r,1),1);
else
    if(Ni==1)
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
title(['3D Plot of dimensionally reduced vectors from ',filename]);
xlabel('x axis');
ylabel('y axis');
zlabel('z axis');
legend(lgdtxt);
view(30,30);



%% Classification using Artificial Neural Network
%-----------------------------------------------------------------------
% Multi-class Classification done using ANN
% No. of nodes in input layer is the dimension of input
% Output has a single node, value outputed denotes the class
% One hidden layer present, with appropriate no. of nodes
%-----------------------------------------------------------------------
%% Neural Network Load
TrainDR(tek,:) = [];
Ni = size(TrainDR,2);  % Number of Input Layer Nodes
No = 1;  % Number of Output Layer Nodes
Ns = size(TrainDR,1);  % Training data sample size
Nt = size(TestDR,1);  % Test data sample size
h1 = ceil(Ns/(3*(Ni+No)));  % Criterion to prevent overfitting
h2 = floor((Ni+No)/2);  % Popular way to get size of hidden layer
h3 = 2*Ni-1;  % Criterion to prevent overfitting
Nh = min([h1,h2,h3]);  % Number of Hidden Layer Nodes


%% Reordering of Training Data
tk = randperm(size(TrainDR,1));
TrainDR = TrainDR(tk,:);
Cls = Cls(tk);


%% Neural Network Initialization
P = zeros(Ni,Nh);  % Input - Hidden Layer
Q = zeros(Nh,No);  % Hidden Layer - Output
u = zeros(1,Nh);  % Hidden Layer bias
v = zeros(1,No);  % Output bias
LR = 0.001;  % Learning Rate
Nepoch = 1000;  % Number of Epochs for training
SSerr = zeros(1,Nepoch);  % Sum Squared Error during each epoch


%% Neural Network Training
for k = 1:1:Nepoch
    for i = 1:1:Ns
        % Forward Propagation of Data
        IP = TrainDR(i,:);
        H = 1./(1+exp(-(IP*P + u)));
        OP = H*Q + v;
        T = Cls(i);  % Target Value
        E = T - OP;  % Error
        SSerr(k) = SSerr(k) + sum(E.^2);  % Sum Squared Error
        
        % Back-propagation and weight adjustment
        Q = Q + LR*(H')*E;
        v = v + LR*E;
        tmp1 = repmat(E*(Q') , [Ni,1]);
        tmp2 = (IP')*(H.*(1-H));
        P = P + LR*tmp1.*tmp2;
        u = u + LR*(H.*(1-H)).*(E*(Q'));
    end
end


%% Sum Squared Error Plot
figure;
plot(1:Nepoch,SSerr);
xlabel('Epochs');
ylabel('Sum Squared Error');
title('Sum Squared Error of Neural Network for each epoch');


%% Model of Trained Neural Network
figure;
hold on;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
axis equal;
title('Model of Trained Neural Network');
pt1 = ( 2*(-(Ni-1)/2:(Ni-1)/2) )';
pt2 = ( 2*(-(Nh-1)/2:(Nh-1)/2) )';
pt3 = ( 2*(-(No-1)/2:(No-1)/2) )';

Ctr1 = [repmat(2,[Ni,1]),pt1];
Ctr2 = [repmat(5,[Nh,1]),pt2];
Ctr3 = [repmat(8,[No,1]),pt3];
Rd1 = repmat(0.2,[Ni,1]);
Rd2 = repmat(0.2,[Nh,1]);
Rd3 = repmat(0.2,[No,1]);
viscircles(Ctr1,Rd1,'EdgeColor','k');
viscircles(Ctr2,Rd2,'EdgeColor','k','LineStyle','--');
viscircles(Ctr3,Rd3,'EdgeColor','k');
plot(Ctr1(:,1),Ctr1(:,2),'.k','LineWidth',2);
plot(Ctr2(:,1),Ctr2(:,2),'.k','LineWidth',2);
plot(Ctr3(:,1),Ctr3(:,2),'.k','LineWidth',2);
for i = 1:Ni
    for j = 1:Nh
        line([Ctr1(i,1),Ctr2(j,1)],[Ctr1(i,2),Ctr2(j,2)]);
        txt = num2str(P(i,j));
        C = (2.5*Ctr1(i,:) + Ctr2(j,:))/3.5;
        text(C(1),C(2),txt);
    end
end
for i = 1:Nh
    for j = 1:No
        line([Ctr2(i,1),Ctr3(j,1)],[Ctr2(i,2),Ctr3(j,2)]);
        txt = num2str(Q(i,j));
        C = (2.5*Ctr2(i,:) + Ctr3(j,:))/3.5;
        text(C(1),C(2),txt);
    end
end
for i = 1:Nh
    txt = num2str(u(i));
    text(Ctr2(i,1)-0.3,Ctr2(i,2)+0.4,txt);
end
for i = 1:No
    line([Ctr3(i,1),Ctr3(i,1)+1],[Ctr3(i,2),Ctr3(i,2)]);
    txt = num2str(v(i));
    text(Ctr3(i,1)-0.3,Ctr3(i,2)+0.4,txt);
end
for i = 1:Ni
    line([Ctr1(i,1)-1,Ctr1(i,1)],[Ctr1(i,2),Ctr1(i,2)]);
end
txt = {'Input','Layer'};
text(Ctr1(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');
txt = {'Hidden','Layer'};
text(Ctr2(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');
txt = {'Output','Layer'};
text(Ctr3(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');


%% Neural Network Testing
Texp = zeros(Nt,1);  % ANN Classification for Testing Data
To = zeros(Nt,1);
for j = 1:1:Nt
    % Input the Test data and get the output class
    Ti = TestDR(j,:);
    H = 1./(1+exp(-(Ti*P + u)));
    To(j) = H*Q + v;
    Texp(j) = round(To(j));  % Output rounded off
end
Accr = sum(Texp==Tecl);  % Accuracy of ANN
disp(['The Neural Network correctly classified ',num2str(Accr),...
    ' out of ',num2str(Nt),', resulting in an accuracy of ',...
    num2str(Accr*100/Nt),' %']);