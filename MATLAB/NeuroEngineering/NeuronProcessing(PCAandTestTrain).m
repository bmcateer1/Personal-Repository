 %% HW 2: Linear regression on neural data
clc  %clears all 
clear all 
close all

load('ori32_M160825_MP027_2016-12-15')

data = stim.resp;
stimlist = stim.istim;

stimuli = unique(stimlist); % istim= stimulus identity of each data, returns all unique stimuli
number = length(stimuli); %states how many unique stimuli there are

zsc = zscore(data); %z-scores the data

%% Step 1: Break data set into training (90%) and testing (10%)
% NOTE USE ONLY 100 neurons because otherwise your ratio of trials to Betas
% is too small!

StimNum = length(data(:,2));  %Total number of stimuli
NeuronNum = length(data(1,:)); %Total Number of Neurons
RandSamp = randi(NeuronNum,100,1);  %Creates a list of 100 random neurons to pick 
LinData = data(:,RandSamp);  %Takes 100 neurons from the random indexing in RandSamp and takes all of their stimuli events

TrainSize = round(0.9 * size(LinData,1)); % 90% of the selected neuron number
TestSize = round(0.1 * size(LinData,1));  % 10% of the selected neuron number

Train = LinData(1:TrainSize,:);        %Seperates 90% of the selected neurons
Test = LinData(TrainSize+1:StimNum,:); %Seperates 10% of the selected neurons

%% Step 2: Find Beta matrix for linear regression between neural responses
% perform regression on TRAIN subset of data 
% [X] = [column of ones , neural response matrix]
% [Y] = list of stimulus presentations
% [B,BINT,R,RINT,STATS] = regress(Y,X)
% X should have each row be a trial (one stimulus presentation)
% X should have each column be a neuron and one column of just ones
% Y is a vector, length equal to the nubmer of trials
% R returns the R^2 value of the fit


Xtrain = [ones(length(Train),1),Train];  %adds a column of ones before the set of train subdata
Ytrain = stimlist(1:TrainSize,:);        %shows matrix of stimuli in order for all Train data subset
[B,BINT,R,RINT,STATS] = regress(Ytrain,Xtrain);  %performs linear regression on the train data for each stimuli event, outputs all major components of linear regression



%% Step 3: Use the Beta matrix you fit in step 2 to predict y values 
% y_hat = X_test * B

X_Test = [ones(length(Test),1),Test];  %adds a column of ones before the set of test data
Y_Hat = X_Test * B;   %creates a predicted stimli value for the test neurons given the training neurons values

%% Step 4: Compute R_sq for test data
 
Y_Test = stimlist(TrainSize+1:StimNum,:);  %Creates a matrix of stimuli in order for all Tested data subset
SS_res = sum((Y_Hat - Y_Test).^2);         %Performs Sum of Squares 
SS_tot = sum((Y_Test - mean(Ytrain)).^2);  %Performs sum of squares of the total
Rsq = 1 - SS_res/SS_tot;                   %Calculates R^2 value from the Sum of Squares


%% Step 5: Plot y_test verses y_hat; Display R^2 value

%Plots the actual test subset neuron stimulus values vs the predicted values
figure (1)
plot(Y_Test,Y_Hat, 'k.')
hold on
plot(Y_Test,Y_Test, 'b-')
xlabel('Y Test')
ylabel('Y Hat')

title(sprintf('Random 100 Neurons Predicted with an R-Squared of %.3f',Rsq))



%% Step 6: In HW1, you calculated the OSI of each neuron.
% Let's assume that neurons with an OSI > 0.4 can be called "well-tuned"
% Repeat steps 1 - 5, only using "well-tuned" neurons in you analysis
% HINT: the only change you need to make is to your X matrix!

stimu = unique(stim.istim); %returns only non-repeated stimulus values
len = length(stimu); %finds length stimuli variable
neurons = size(data,2); %sizes the number of neurons tested

for t = 1:len %iterates through every stimulus
    tuningcurve(t,:) = mean(data(stim.istim == stimu(t),:)); %calculates a response for each neuron 
end

for t = 1:neurons
    maxtune(t) = max(tuningcurve(:,t)); %finds the max activity for each neuron
    orient(t) = find(tuningcurve(:,t) == maxtune(t));% finds where the max activity for each neuron matches in orientation to the stimuli direction 
end

for t = 1:neurons
    direction = tuningcurve(:,t); 
    %Resp_min = min(tuningcurve(:,t)); %finds the minimum response for each
    %neuron NOT USED IN THIS FORMULA
    Resp_PD =  max(tuningcurve(:,t)); %finds the maximum response for each neuron
    
    %here, assuming all 33 directions cover a full 360 degrees, if a neuron
    %were to have an orthogonal value outside of the range 1:33, a
    %correction is applied to keep its value at around 90 degrees from the
    %response
    if (find(direction == Resp_PD) - 8) <= 0 
        iLow = (find(direction == Resp_PD) - 8) + 33;
    else 
        iLow = (find(direction == Resp_PD) - 8);
    end
    
    if (find(direction == Resp_PD) + 8) >= 34
        iHigh = (find(direction == Resp_PD) + 8) - 33;
    else
        iHigh = (find(direction == Resp_PD) + 8);
    end
    
    Resp_orthLow = direction(iLow); %the orthogonal response lower than the regular
    Resp_orthHigh = direction(iHigh); %the orthogonal response higher than the regular
    Resp_orth = min([Resp_orthLow,Resp_orthHigh]); %chooses whichever orthoganal angle is lower
    
    OSI(t) = (Resp_PD - Resp_orth)/(Resp_PD + Resp_orth); %performs OSI calculation
end

TunedNeurons = find(OSI > 0.4);  %Finds all neurons with an OSI greater than 0.4 and catches their index

TunedNum = length(TunedNeurons);  %determines the number of tuned neurons with an OSI greater than 0.4

RandTunedSamp = randi(TunedNum,100,1); %grabs a random sample of 100 neurons from the tuned neuron subset
TunedLinData = data(:,TunedNeurons); %Indexes 100 neurons from the subset of tuned neurons and takes all of their stimuli event data
TunedLinData = TunedLinData(:,RandTunedSamp);

TTrain = TunedLinData(1:TrainSize,:); %takes 90% of the stimuli responses from the tuned neuron subset to predict last 10%
TTest = TunedLinData(TrainSize+1:StimNum,:); %takes 10% of the stimuli responses from the tuned neuron subset to test prediction

TXset = [ones(length(TTrain),1),TTrain]; %Adds ones column to 90% of the stimulus responses from the tuned neuron subset to predict last 10%
TYset = stimlist(1:TrainSize,:); %Adds ones column to other 10% of the stimulus responses from the tuned neuron subset to test prediction
[BT,BINTT,RT,RINTT,STATST] = regress(TYset,TXset);  %performs linear regression on the tuned train data for each stimuli event, outputs all major components of linear regression

TX_Test = [ones(length(TTest),1),TTest]; %Adds ones column to 10% of the stimulus responses from the tuned neuron subset to test prediction
TY_Hat = TX_Test * BT; %predicted stimuli for the last 10% of stimuli

TY_Test = stimlist(TrainSize+1:StimNum,:); %creates an array of last 10% of stimuli from tuned neuron subset

TSS_res = sum((TY_Hat - TY_Test).^2);      %Performs sum of squares 
TSS_tot = sum((TY_Test - mean(TYset)).^2); %Performs sum of squares of the total
TRsq = 1 - TSS_res/TSS_tot;                %Calculates R^2 of the tuned neuron total

figure (2) %Plots the actual data vs the predicted stimuli for the last 10% of the neurons
plot(TY_Test,TY_Hat, 'k.')
xlabel('Y Test')
ylabel('Y Hat')
hold on
plot(TY_Test,TY_Test, 'b-')
title(sprintf('Random 100 Well-Tuned Neurons with an R-Squared of %.3f',TRsq))

%% Step 7: Perform PCA on neural response data
% [COEFF, SCORE, LATENT] = pca(X); <-- here X is the full dataset (all
% 11000 neurons)
% SCORE is the T matrix from lecture: the data in PCA space
% Repeat steps 1 -5 using first 100 PC (columns 1-100 of SCORE) instead of
% the raw responses.
% HINT: Again, you are just working to change X; first, change the axes you
% use for X by doing PCA. Then, grab the first 100 PCs, and repeat linear
% regression.


%StimNum = length(data(:,2));  %Total number of stimuli
%NeuronNum = length(data(1,:)); %Total Number of Neurons
%RandSamp = randi(NeuronNum,100,1);  %Creates a list of 100 random neurons to pick 
%LinData = data(:,RandSamp);  %Takes 100 neurons from the random indexing in RandSamp and takes all of their stimuli events
[COEFF, SCORE, LATENT] = pca(data);  %Performs PCA on data

TrainPCA = SCORE(1:TrainSize,1:100);
TestPCA = SCORE(TrainSize+1:StimNum,1:100);

X_PCA = [ones(length(TrainPCA),1),TrainPCA];  %adds a column of ones before the set of PCA train subdata
%Ytrain = stimlist(1:TrainSize,:);        %shows matrix of stimuli in order for all Train data subset
[Bpca,BINTpca,Rpca,RINTpca,STATSpca] = regress(Ytrain,X_PCA);  %performs linear regression on the train data for each stimuli event, outputs all major components of linear regression

PCA_X_Test = [ones(length(TestPCA),1),TestPCA];  %adds a column of ones before the set of test data
PCA_Y_Hat = PCA_X_Test * Bpca;   %creates a predicted stimli value for the test neurons given the training neurons values

PCA_Y_Test = stimlist(TrainSize+1:StimNum,:);  %Creates a matrix of stimuli in order for all Tested data subset
PCA_SS_res = sum((PCA_Y_Hat - PCA_Y_Test).^2);         %Performs Sum of Squares 
PCA_SS_tot = sum((PCA_Y_Test - mean(Ytrain)).^2);  %Performs sum of squares of the total
Rsq = 1 - PCA_SS_res/PCA_SS_tot;                   %Calculates R^2 value from the Sum of Squares

figure (3)
plot(PCA_Y_Test,PCA_Y_Hat, 'k.')
hold on
plot(PCA_Y_Test,PCA_Y_Test, 'b-')
xlabel('Y Test')
ylabel('Y Hat')
title(sprintf('Top 100 PCs with an R-Squared of %.3f',Rsq))