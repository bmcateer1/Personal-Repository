function [] = Problem_1A

%% Total Function with all differential equations
HR = 80; %heart rate (in beats/min)

Ra = 0.2*HR;    %mid aorta dividing proximal and distal segment resistance (in mmHg /( L * min))
Rp1 = 36*HR;   %peripheral and vascular resistance of upper body (in mmHg /(L * min))
Rp2 = 36*HR;   %peripheral and vascular resistance of lower body (in mmHg / (L * min))
Rin = 0.2*HR;   %input  resistance for the heart pump (in mmHg / (L * min))
Rout = 0.2*HR; %output valve resistance (in mmHg / (L * min))
St = 0.2*HR; %resistance with double pump on

Ca1_0 = 0.00052; %upper aorta compliance (in L/(beat*mmHg)
Ca2_0 = 0.00052; %lower aorta compliance (in L/(beat*mmHg)
Cv = 0.0312;     %center veins compliance  (in L/(beat*mmHg)
Cp = 0.0125;    %heart pump compliance  (in L/(beat*mmHg)
P0 = 100; %Variable Compliances (in mmHg)

w = 8.378; %(in Hz)
SV = 0.0625; %stroke volume(in mmHg)
Vmax = 0.07; %max volume (in L) 
Delta_Pmax = 80; %peak left ventricular compression pressure(in mmHg)
phi = 3.14; 
phi_s = 5.5; 
A = 0.6; %trigger value

T0 = 0; %start time in sec
Tf = 10; %end time in sec
dt = 0.000001; %time step in sec
Num_step = (Tf/dt)+1; %Number of time points
 
%% Initializing arrays 
B = zeros(1,Num_step);  %initilzing empty array for each array
Pp = zeros(1,Num_step);
Pa1 = zeros(1,Num_step);
Pa2 = zeros(1, Num_step);
Pv = zeros(1, Num_step); 
Pow = zeros(1, Num_step); 

Pp_off = zeros(1,Num_step);
Pa1_off = zeros(1,Num_step);
Pa2_off = zeros(1, Num_step);
Pv_off = zeros(1, Num_step); 

t = T0:dt:Tf; %time step array

B_dub = zeros(1,Num_step);
Pp_dub = zeros(1,Num_step);
Pa1_dub = zeros(1,Num_step);
Pa2_dub = zeros(1,Num_step);
Pv_dub = zeros(1,Num_step);
Pow_dub = zeros(1, Num_step); 

%% Initial Pressure and Power Values
Pa1_off(1)=10; 
Pa2_off(1)=10; 
Pp_off(1)=10;  
Pv_off(1)=10; 

Pa1(1)=10; %initial upper body pressure for a single pump(in mmHg)
Pa2(1)=10; %initial lower body pressure
Pp(1)=10;  %initial heart pump pressure
Pv(1)=10;  %initial vein pressure
Pow(1)=0; %left ventricle power

Pa1_dub(1)=10; %same as above but for double pump
Pa2_dub(1)=10;
Pp_dub(1)=10;
Pv_dub(1)=10;
Pow_dub(1)=0;

%% Double Pump Pressures, Flows, and Power
for count = 2:1:Num_step %creates a for loop to calculate each pressure per fraction of a second
    c2 = count-1;
    if cos(w*t(count))>0
        Pext = (Delta_Pmax /2 *sin(w*t(count))*w);
    else
        Pext = 0;
    end
    B_dub(count) = Vmax / 2 * (w * sin(w * t(count)+phi));
    Ca1 = Ca1_0*P0/Pa1_dub(c2);
    Ca2 = Ca2_0*P0/Pa2_dub(c2);

     if sin(w*t(c2)-phi_s)
         St =  max(0,1000 * (1 / (1 - A) * cos(w * t(c2) - phi_s)) - A);
     else
         St = 0;
     end
     
    Pp_dub(count) = Pp_dub(c2) + (Pext + (1/Cp)*((max(0,Pv_dub(c2)-Pp_dub(c2))/Rin)-(max(0,Pp_dub(c2)-Pa1_dub(c2))/Rout)))*dt;
    Pa1_dub(count) = Pa1_dub(c2) + ( (1/Ca1)*(B_dub(count)+((max(0,Pp_dub(c2)-Pa1_dub(c2))/Rout)-((Pa1_dub(c2)-Pa2_dub(c2))/(Ra + Ra * St) + (Pa1_dub(c2)-Pv_dub(c2))/Rp1))))*dt;
    Pa2_dub(count) = Pa2_dub(c2) + ((1/Ca2)*((Pa1_dub(c2)-Pa2_dub(c2)) / (Ra+Ra*St) - (Pa2_dub(c2)-Pv_dub(c2))/Rp2))*dt;
    Pv_dub(count) = Pv_dub(c2) + ((1/Cv)*((Pa1_dub(c2)-Pv_dub(c2))/Rp1 + (Pa2_dub(c2)-Pv_dub(c2))/Rp2 - (max(0,Pv_dub(c2)-Pp_dub(c2)))/Rin))*dt;
    Pow_dub(count) = max(0,(SV * (((Pa1_dub(c2) + Pa2_dub(c2)) / 2) - Pv_dub(c2)) / Rout * (Pa1_dub(c2) - Pv_dub(c2)))); 
   
end

%% Single Pump Pressures, Flows, and Power
for count = 2:1:Num_step %creates a for loop to calculate each pressure per fraction of a second
    c2 = count-1;
    if cos(w*t(count))>0
        Pext = (Delta_Pmax/2*sin(w*t(count))*w);
    else
        Pext = 0;
    end
    B(count) = Vmax / 2 * (w * sin(w * t(count)+phi));
    Ca1 = Ca1_0*P0/Pa1(c2);
    Ca2 = Ca2_0*P0/Pa2(c2);

    Pp(count) = Pp(c2) + (Pext + (1/Cp)*((max(0,Pv(c2)-Pp(c2))/Rin)-(max(0,Pp(c2)-Pa1(c2))/Rout)))*dt;
    Pa1(count) = Pa1(c2) + ( (1/Ca1)*(B(count)+((max(0,Pp(c2)-Pa1(c2))/Rout)-((Pa1(c2)-Pa2(c2))/Ra + (Pa1(c2)-Pv(c2))/Rp1))))*dt;
    Pa2(count) = Pa2(c2) + ((1/Ca2)*((Pa1(c2)-Pa2(c2))/ Ra - (Pa2(c2)-Pv(c2))/Rp2))*dt;
    Pv(count) = Pv(c2) + ((1/Cv)*((Pa1(c2)-Pv(c2))/Rp1 + (Pa2(c2)-Pv(c2))/Rp2 - (max(0,Pv(c2)-Pp(c2)))/Rin))*dt;
    Pow(count) = max(0,SV * (((Pa1(c2) + Pa2(c2)) / 2) - Pv(c2)) / Rout * (Pa1(c2) - (Pv(c2))));
     
end

%% Pump off Pressures and Flows
for count = 2:1:Num_step  %creates a for loop to determine pressures when Vmax = 0 or when the heart pump is turned off
    c2 = count-1;
    if cos(w*t(count))>0
        Pext = (Delta_Pmax*sin(w*t(count))*w);
    else
        Pext = 0;
    end
    Ca1 = Ca1_0*P0/Pa1(c2);
    Ca2 = Ca2_0*P0/Pa2(c2);
  
    Pp_off(count) = Pp_off(c2) + (Pext + (1/Cp)*((max(0,Pv_off(c2)-Pp_off(c2))/Rin)-(max(0,Pp_off(c2)-Pa1_off(c2))/Rout)))*dt;
    Pa1_off(count) = Pa1_off(c2) + (1/Ca1)*((max(0,Pp_off(c2)-Pa1_off(c2))/Rout)-((Pa1_off(c2)-Pa2_off(c2))/Ra + (Pa1_off(c2)-Pv_off(c2))/Rp1))*dt;
    Pa2_off(count) = Pa2_off(c2) + ((1/Ca2)*((Pa1_off(c2)-Pa2_off(c2))/Ra - (Pa2_off(c2)-Pv_off(c2))/Rp2))*dt;
    Pv_off(count) = Pv_off(c2) + ((1/Cv)*((Pa1_off(c2)-Pv_off(c2))/Rp1 + (Pa2_off(c2)-Pv_off(c2))/Rp2 - (max(0,Pv_off(c2)-Pp_off(c2)))/Rin))*dt;

end

%% Plots and Graphs

figure %plots power of left ventricle with single and double pump
hold on
plot(t,Pow, 'r')
plot(t,Pow_dub, 'b')
legend('Single pump','Double Pump')
xlabel('Time(sec)')
ylabel('Power (W)')
title('Ventricular Power')

figure %plots the contrasting results of when the pump is off and when it is on with factored resistances
hold on
grid on
%plot(t,Pa1,'r')
%plot(t,Pa2, 'g')
plot(t,Pa1_off,'b')
plot(t,Pa2_off, 'c')
%'Single Pump Thorax','Single Pump Abdomen',
plot(t,Pa1_dub, 'm')
plot(t,Pa2_dub, 'k')
legend({ 'Pump off Thoracic','Pump off Abdominal','Double Pump Thoracic', 'Double Pump Abdominal'},'Location','Southeast')
xlabel('Time(sec)')
ylabel('Pressure(mmHg)')
title('Pressures with Double Pump On and Off')


figure %plots power of left ventricle with single and double pump
hold on
plot(t,Pow, 'r')
plot(t,Pow_dub, 'b')
legend('Single pump','Double Pump')
xlabel('Time(sec)')
ylabel('Power (W)')
title('Ventricular Power')


figure %plots each different pressure 
subplot (2,2,1)
plot(t,Pp)
title('Pp Heart Pump')
xlabel('Time(in sec)')
ylabel('Pressure (mmHg)')
subplot (2,2,2)
plot(t,Pa1)
ylabel('Pressure (mmHg)')
xlabel('Time (in secs)')
title('Pa1 Upper body')
subplot(2,2,3)
plot(t,Pv)
ylabel('Pressure (mmHg)')
xlabel('Time (in secs)')
title('Pv Veins pressure')
subplot(2,2,4)
plot(t,Pa2)
ylabel('Pressure (mmHg)')
xlabel('Time (in secs)')
title('Pa2 Lower Body')
   
end



