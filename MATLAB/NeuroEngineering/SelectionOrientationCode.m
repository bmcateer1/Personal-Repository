clc
clear 
clear all %clears all previous data and plots

%%  1. load ori32_M170714_MP032_2017-08-02.mat
load('ori32_M160825_MP027_2016-12-15.mat') %33 movement directions reactions of neurons

%%  2. get responses from stim variable (stim.resp), which is ncell x nresp

figure(1) %plots raw data
imagesc(stim.resp)
colorbar %adds a colorbar
xlabel('Neuron #')
ylabel('Stimulus Presentation #')

data = stim.resp; %makes a variable name for the stimulus response

%%  3. z-score responses: subtract mean and divide by standard deviation

zsc = zscore(data); %z-scores the data

figure(2)
imagesc(zsc') %transposes data and plots the z-scored data
colorbar %adds a colorbar
xlabel('Stimulus')
ylabel('Neuron #')
title('Z-scored Data')

%%  4. Compute tuning curves
stimu = unique(stim.istim); %returns only non-repeated stimulus values
len = length(stimu); %finds length stimuli variable
neurons = size(data,2); %sizes the number of neurons tested

for t = 1:len %iterates through every stimulus
    tuningcurve(t,:) = mean(data(stim.istim == stimu(t),:)); %calculates a response for each neuron 
    ztune(t,:) = mean(zsc(stim.istim == stimu(t),:)); %calculates response for z-scaled neurons
end

%%  5. Plot tuning curves

figure(3) %plots the z-scaled and regular plots together
subplot(2,1,1)
imagesc(tuningcurve)
xlabel('Neuron #')
ylabel('Mean Response')
title('Tuning Curves')
subplot(2,1,2)
imagesc(ztune)
xlabel('Neuron #')
ylabel('Mean Response')
title('Tuning Curves of Z-scored')

n = 49; %arbritary number of plots to subplot
subn = ceil(rand(n,1)*neurons); %plots random subset of neurons each time
dimen = ceil(sqrt(length(subn))); 

figure (4)
for t = 1:length(subn) %plots the random neuron response graphs
    subplot(dimen,dimen,t)
    plot(tuningcurve(:,subn(t)))
    title(sprintf('# %d',subn(t)))
    xlabel('Stimulus #')
    ylabel('Mean Response')
end

figure (5)
for t = 1:length(subn)%plots the random z-scaled neuron response graphs 
    subplot(dimen,dimen,t)
    plot(ztune(:,subn(t)))
    title(sprintf('# %d',subn(t)))
    xlabel('Stimulus #')
    ylabel('Mean Response')
    axis([0 34 -0.5 3]) %zooms graph into number of stimuli
end

%% 6. Compute Preferred direction for each neuron (direction of maximum activity)

for t = 1:neurons
    maxtune(t) = max(tuningcurve(:,t)); %finds the max activity for each neuron
    orient(t) = find(tuningcurve(:,t) == maxtune(t));% finds where the max activity for each neuron matches in orientation to the stimuli direction
    
end

%%  7. Plot PDs for population of neurons (make a histogram of PDs)

figure (6)
histogram (orient) %plots the prefered orientation of each neuron together
xlabel('Direction')
ylabel('Frequency')
xlim([0 34])
title('Prefered Direction of Each Neuron')

%%  8. Compute orientation selectivity index for each neuron (OSI)
%  OSI = (Resp_PD - Resp_orth)/(Resp_PD + Resp_min)

for t = 1:neurons
    direction = tuningcurve(:,t); 
    Resp_min = min(tuningcurve(:,t)); %finds the minimum response for each neuron
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
    
    OSI(t) = (Resp_PD - Resp_orth)/(Resp_PD + Resp_min); %performs OSI calculation
end

%%  9. Plot orientation selectivity for population of neurons (make a histogram....)

figure (7)
histogram (OSI) %Graphs the OSI 
xlabel('Sensitivity')
ylabel('Frequency of Neurons')
title('Orientation selectivity index')

