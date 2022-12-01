function[] = Problem_2

%% Variable Initialization 
R_un = 78;% microns * 10^-6; %R0 in word doc, radius of unstressed microbubble(in meters) 
Lconvert = 10^-6; %microns to meters convert
Tconvert = 10^6; %seconds to microseconds
rho = 1000;  %Density (in kg/m^3)
Pconvert = 133; %mmHg to kg/(m*s^2)conversion
P1 = 760; % atmospheric pressure (enter in mmhg, converted to Pa or kg/(m*s^2) with Pconvert) 
Pmax = 50; % Max Pressure of right ventricle (in mmHg)
T0 = 0; %start time in sec
Tf = 5000; %end time in microseconds
dt = 0.01; %time step in sec
Num_step = (Tf/dt)+1; %Number of time points
Delta_P = 0; %initial change in pressure
kHz = 1/1000; %Hz to kHz
Temp = 2; %Change in Temperature in degrees Celcius
Tvar = 0.308 * Temp * Pconvert; %Pressure variation in mmHg
b = 6907.76; %damping factor (unitless)

Avg_P = [0,17,22,27,35,45]; %AVG pressures given in #9 mmHg
Avg_Freq = [34.2,35.9,36.1,36.2,36.0,36.7]; %AVG Frequency given in #9 in kHz

Avg = 0;
SDev = 0;

St =0; 
Ft = 1000000; %time per cycle times cycles (in microseconds)
Dt = 10;
T_step = (Ft / Dt) + 1;
%% Function initialization
t = T0:dt:Tf; %time step array
ti = St:Dt:Ft;

R = zeros(St,Ft); %Empty Array of change in radius
pmax = zeros(St,Ft);
Soundtrack = zeros(1,Ft);
Soundtrack2 = zeros(1,Ft);
Z = zeros(1,Ft);

Po = zeros(0,Pmax); %Empty Array Right Ventricular Pressure
Freq = zeros(0,Pmax); %Empty Array of Popping Frequency
Po2 = zeros(0,Pmax); %Empty Array Right Ventricular Pressure
Freq2 = zeros(0,Pmax); %Empty Array of Popping Frequency
Freq_change = zeros(0,Pmax); %Empty Array of Popping Frequency


%% Initial Function Values
R(1) = 0.2; %radius of compressed sugar coated microbubble (in microns)

Po(1) = 760; %* Pconvert; % Initial Right Ventricular Pressure
Freq(1) = 34.2; %Initial Frequency of Microbubbles
Freq_change(1) = 0;

Po2(1) = Po(1);
Freq2(1) = 34.2;
R_un2 = R_un; %Needs changed 
P1_2 = P1 + Tvar;

%% Radius of Microbubbles
for count = 2:1:T_step %creates a for loop to calculate each pressure per fraction of a second
    c2 = count-1;
   
    Po(count) = Delta_P - 1 + P1; %Pressure (Delta P * Pconvert 
    w = sqrt((3 * Po(c2)) / (R_un * Lconvert * rho))  * 6; % Omega
   
    R(count) = cos(w * ti(c2) *1.5 ) * expm(-b * ti(c2) * 3*10^-5 / 12); %change in radius per cycle  
end

%% Monte Carlo Simulation
new = 0;
t=0;
count = 1;
Wcount = 0;
c2 = 1;
while t < 1000000 && count < 100000000 
    
    if t >= new
        loc_p = rand(); % location probability of bubble popping in heart

        if loc_p <= 0.3
            loc = ('Right Atrium');
            Ps = 2; %systolic pressure right atrium (in mmHg) TA Tyler told us systolic pressure was lower than diastolic in right atrium
            Pd = 6; %diastolic
        elseif loc_p <=0.7 
            loc = ('Right Ventricle');
            Ps = 30; %right ventricle
            Pd = 5; %diastolic
        elseif loc_p <=1.0
            loc = ('Pulmonary Artery');
            Ps = 29; %systolic pulmonary artery pressure (in mmHg)
            Pd = 10; %diastolic pressure
        end
        pmax = rand();
        R_un = normrnd(78,78 * 0.005); %radius with a tolerance of 0.5%, 1%, 5 %
        ws = sqrt((3 * (P1 + Ps) * Pconvert) / (R_un^2 * Lconvert * rho)); % Omega (in Hz)
        wd = sqrt((3 * (P1 + Pd) * Pconvert) / (R_un^2 * Lconvert * rho)); % Omega (in Hz)
        W = normrnd(1000,300); % in micros actually a w
        new = t+W;
        Wcount = Wcount + 1; %Counts number of bubbles created
    end  
    
    for tprime = 0:floor(W)
        if t < 1000000 && count < 100000000
        Soundtrack(count) =  pmax * exp(-b * tprime * 3*10^-6 ) * sin(ws * t / 1000); %Systolic Relative Sound Pressure
        Soundtrack2(count) = pmax * exp(-b * tprime * 3*10^-6 ) * sin(wd * t / 1000); %Diastolic Relative Sound Pressure
        Z(count) = t; 
        count = count+1;
        t = t + 1; 
        end
    end
end

tstar = zeros(1,10001); %initialized time between systolic time tracks
tstard = zeros(1,10001); %initialized time between diastolic time tracks
for c2 =1:10001
    if (Soundtrack(c2) > 0 && Soundtrack(c2+1) < 0) || (Soundtrack(c2) < 0 && Soundtrack(c2+1) > 0) 
    tstar(c2) = ((Soundtrack(c2) * Z(c2+1)) - (Z(c2) * Soundtrack(c2+1))) / (Soundtrack(c2)-Soundtrack(c2+1));%Determines when systolic soundtrack crosses zero 
    end
    if (Soundtrack2(c2) > 0 && Soundtrack2(c2+1) < 0) || (Soundtrack2(c2) < 0 && Soundtrack2(c2+1) > 0) 
    tstard(c2) = ((Soundtrack2(c2) * Z(c2+1)) - (Z(c2) * Soundtrack2(c2+1))) / (Soundtrack2(c2)-Soundtrack2(c2+1));%Determines when diastolic soundtrack crosses zero 
    end
end

c4 = 1;
c11 = 1;
tstar_final = zeros(1,nnz(tstar)); %creates an array of nonzero values for the moment before soundtrack crossses zero
tstar_finald = zeros(1,nnz(tstar));
for c3 = 1:10001
    if tstar(c3) ~= 0 %removes zeros from the tstar array
        tstar_final(c4) = tstar(c3);
        c4 = c4+1;
    end
    if tstard(c3) ~= 0 
        tstar_finald(c11) = tstard(c3);
        c11 = c11+1;
    end
end
P_m = zeros(1,length(tstar_final)-1); %measured pressure
P_m(1) = Ps; 

iFreq = zeros(1,length(tstar_final)-1); %initiallizes instantaneous frequencies
iFreq2 = zeros(1,length(tstar_finald)-1);

for c5 = 1:(length(tstar_final)-1)
    iFreq(c5) = Tconvert / (2 * (tstar_final(c5+1) - tstar_final(c5))); %instantaneous frequency of systolic
end

for c5 = 1:(length(tstar_final)-1)
    iFreq2(c5) = Tconvert / (2 * (tstar_finald(c5+1) - tstar_finald(c5))); %instantaneous frequency of diastolic
end
%instantaneous frequency statistics
iFreqavg = mean(iFreq);
iFreqstd = std(iFreq);
iFreqmode = mode(floor(iFreq));
iFreqavg2 = mean(iFreq2);
iFreqstd2 = std(iFreq2);
iFreqmode2 = mode(floor(iFreq2));

for c5 = 1:(length(tstar_final)-1)
    if (iFreq(c5) < 34000) || (iFreq(c5) > 37000)  
         iFreq(c5) = 0; %removes outlier data with a previous data point, zooms in towards the ideal frequency
    end
    if (iFreq2(c5) < 34000 ) || (iFreq2(c5) > 37000 ) 
         iFreq2(c5) = 0; %removes outlier data with a previous data point, zooms in towards the ideal frequency
    end
end

lengthFreq = length(iFreq2);
c4 = 1;
for c3 = 1:length(iFreq)
    if iFreq(c3) ~= 0 
        iFreqfinal(c4) = iFreq(c3); %removes outlier of zeros of frequency for systole
        c4 = c4+1;
    end
end
c4 = 1;
for c3 = 1:length(iFreq2)
    if iFreq2(c3) ~= 0 
        iFreq2final(c4) = iFreq2(c3); %removes outlier of zeros of frequency for diastole
        c4 = c4+1;
    end
end

for c5 = 1:(length(tstar_final)-2)
    P_m(c5+1) = (2 * (iFreq(c5+1) - iFreq(c5)) / iFreq(c5) - 1) * P_m(c5); %calculated artierial -Pressure for whichever area in the heart.
end

Pavg = mean(P_m);
Pstd = std(P_m);

%% Frequency of Microbubbles and Frequency Change

for Delta_P = 1:Pmax %creates  a for loop for frequenct
    
    Po(Delta_P) = Delta_P * Pconvert + P1; %Po in Pa, Delta
    w = sqrt((3 * Po(Delta_P)) / (R_un^2 * rho)); %Omega
    Freq(Delta_P) = w / (2 * 3.14) * kHz;
    
    Po2(Delta_P) = Delta_P * Pconvert + P1_2;
    w2 = sqrt((3 * Po2(Delta_P)) / (R_un2^2 * rho)); %Omega
    Freq2(Delta_P) = w2 / (2 * 3.14) * kHz;
    
    Freq_change(Delta_P) = (Freq2(Delta_P) - Freq(Delta_P)) / Freq(Delta_P) * 100;
end
Delta_P = (1:Pmax);


%% Results 

figure 
hold on
histogram(iFreqfinal,30)
title('Optimal Systolic Frequency Distribution')
xlabel('Frequency (Hz)')
ylabel('Occurrences (1000ms)')

figure
hold on
histogram(iFreq2final,30)
title('Optimal Diastolic Frequency Distribution')
xlabel('Frequency (Hz)')
ylabel('Occurrences (1000ms)')

%{
figure
hold on
plot(Z,Soundtrack)
title('Distortion of Bubble Frequency Over Time')
xlabel('Time (microseconds)')
ylabel('Sound Pressure (mmHg)') 

figure
hold on 
plot(ti,R)
xlabel('times')
ylabel('Change in Radius')
title('Radius change over time') 

figure
hold on
plot(Delta_P,Freq)
scatter(Avg_P,Avg_Freq)
title('Frequency and Pressure Comparison')
ylabel('Frequency (in kHz)')
xlabel('Delta P (in mmHg)')

figure
hold on
plot(Delta_P,Freq2)
title('New Frequency and Pressure')
ylabel('Frequency')
xlabel('Delta P')

figure
hold on
plot(Delta_P,Freq_change)
title('Frequency Change')
xlabel('Delta Pressure')
ylabel('Change (in %)')
 
figure
hold on
histogram(pmax,30)
xlabel('Data')
ylabel('Frequency')
title('Histogram of Uniform Distribution')
%}


end