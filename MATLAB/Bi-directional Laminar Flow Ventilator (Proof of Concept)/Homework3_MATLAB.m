
%% Problem 3 
%% Initialization
tic
close all

mu = 0.00017;                   %Viscosity of air at 40 degrees C (g/cm*sec)
rho = 0.0011;                    %Density of air at 40 degrees C (g/cm^3)
Rmax = 0.9;                     %radius of the airway (cm)
L = 20;                         %cm
N = 100;                        %Number of fluid layers in model
Pmax = 52;                      %Max input pressure (dynes/cm^2) **1 dyne/um^2 = 0.1 Pa**
Clung = 0.15;                   %Lung Chest compliance (cm^3/(dynes/cm^2))
f = 2;                          %frequency Hz         
w = 2*pi*f;                     %Angular velocity of the jet radius fraction of Rmax

dr = Rmax /N;                   %delta r
dt = 0.00001;                   %delta t
t1 = 1;                         %starting time for plotting
tf = 4;                         %final time
skip = 10000;                   %for plotting less values and ceating a quicker run time
                   
r = 0:dr:Rmax;              %initalizing radius array
v = zeros(1,N+1);           %initalizing velocity array
flowsum = zeros(1,N+1);     %totals the amount of flow
avgflowsum_store = zeros(26,N+1);
avgflowsum = zeros(1,N+1);
absavgflowsum_store= zeros(26,N+1);
bcflowsum = zeros(1,N+1);

jrf = 0.55;                 %Jet Radius Fraction (Hz) initally 0.33
dv_dt = v;                  %Change in velocity
Plung = 0;                  %initial Pressure of the lung
z=2;                        %count variables
x=1;

%% Layer Calculation  
for t = 0:dt:tf            %time loop  
    z = z+1;
    
    for  n= 1:N            %cumulative lung pressure
        Plung = Plung +(1/Clung)*6.28*r(n)*dr*v(n)*dt; 
    end
    
    dP = (Pmax/2)*(1-cos(w*t)); %change in pressure per time step (dynes/cm^2)
    
    for n = 2:N-1          %radius loop
        dv_dr = (v(n+1)-v(n-1))/(2*dr); 
        d2v_dr2 = (v(n+1)+v(n-1)-2*v(n))/dr^2;
                 
        if ((n/100) > jrf)%jet radius fraction
            dv_dt(n) = (mu / rho)*(d2v_dr2+dv_dr/r(n))-(Plung/(L * rho));
        else
            dv_dt(n) = (mu / rho) *( d2v_dr2 + dv_dr/ r(n)) + (dP-  Plung)/(L * rho);
        end
     end
       
     v = v + dv_dt*dt; 
     v(end) = 0;     %velocity at the edge is zero
     v(1) = v(2);
              
     flowsum(1) = flowsum (1) + pi*(dr/2)^2*v(1);
     for n = 2:N
         flowsum(n) = flowsum(n) + 2*pi*r(n)*dr*v(n);
         %if flowsum(n-1)>=0 && flowsum(n)<=0 %determine if
         %given parameters cause laminar flow without using a
         %plotting function
         %    fprintf('flow sum zero at n = %f \n',n)
         %end
     end
            %size(flowsum)
        
        %plot
      if ((mod(floor(t/dt), (skip+1)) == 0) && (floor(t/dt)*dt > t1))
          x=x+1;
          avgflowsum_store(x,:) = (avgflowsum_store(x-1,:)+flowsum)./(3/dt);
          absavgflowsum_store(x,:) = (absavgflowsum_store(x-1,:)+ abs(flowsum))/(3/dt);
 
          %{
           figure(1)
            hold on
            plot(r, v)
            title('Instantaneous Velocity over Radius')
            xlim([0 .9])
            xlabel('Radius(cm)')
            ylim([-200 400])
            ylabel('Velocity(cm/sec)')
            drawnow
           if v(1)>350
            fprintf('max v  = %f \n',v(1))
           end 
          
            figure(2)
            hold on
            plot(r,flowsum)
            title('Instantaneous Flow per Radius')
            xlabel('Radius(cm)')
            xlim([0 .9])
            ylabel('Flow Sum(cm^3/sec)')
            %ylim([-4*10^5 4*10^5])
 %}        
       end   
end


%end

for x = 1:101
    avgflowsum(x) = sum(avgflowsum_store(:,x))./26; 
    bcflowsum(x) = sum(absavgflowsum_store(:,x))./26;
end
%{
%Plots the average flow per node
figure(3)
            plot(1:101,avgflowsum)
            title('Average Flow per Node')
            xlabel('Node')
            xlim([0 100])
            ylabel('Flow Sum(cm^3/sec)')
            %ylim([-4*10^5 4*10^5])
 %}     

%% Merit
m_pos=0;
m_neg=0;

for x = 1:51
    m_pos = m_pos+bcflowsum(x);
    m_neg = m_neg+bcflowsum(x+50);
end
merit_pos = 0.5*m_pos/51
merit_neg = 0.5*m_neg/51    
merit = sum(bcflowsum)/2              %merit of flow


%{
jrfplot = [0,.1,.2,.3,.4,.5,.55,.6,.7,.8,.9,1];
meritplot = [0,2.4487,9.7504,20.1057,30.7103,38.1646,39.6715,39.2556,32.5687,19.9822,6.4697,0.291];


figure %plots the merit per given jet radius fraction
hold on
plot(jrfplot,meritplot)
xlabel('Jet Radius Fraction (Hz)')
ylabel('Merit (cm^3/s)')
title('Merit per Jet Radius Fraction')
    
%}
toc
