function main
%This function plots the PR EOS and calculates equilibrium

%PR constants for methane       %  Units
omega = 0.0115;             %  dimensionless
Tc = 191.15;                    %  Kelvin
Pc = 4.641;                     %  MPa
Vc = 0.099;                     %  m^3/kmol
R = 8.31413*(10^-3);            %  MPa m^3/(kmol*K)
T_min = 75;                     %  Minimum temperature value

% T_min = 195;  
T_max = 235;                    %  Maximum temperature value
increment = 5;                  %  Increment for isotherms

%Initialize V (I use a log scale to capture events near origin)

for i = 1:8000
     V(i) = 2.8*10^(-2 + i/1000);
end

%Find the dimensions of V for indexing and taking derivatives later
[~,n] = size(V);

%Initialize Temperature range for isotherms.
T_data = T_min:increment:T_max;

%For flexibility, determine the size of T
[~,n1] = size(T_data);

for j = 1:n1
    T = T_data(j);
    
    Tr = T/Tc;              %Calc. Tr

    a = 0.45724*(R^2*Tc^2/Pc);  %PR constants
    b = (0.07780*R*Tc)/Pc;      
    alpha = (1+(0.37464 + 1.54226*omega - 0.26992*omega^2)*(1-(Tr^0.5)))^2;
    
    for i = 1:n             %Calculate pressure by calling function (at bottom)
        P(i) = PR_EOS(a,b,alpha,R,T,V(i));
    end
    
    PV_data = [P;V];
    
    dP_dV_data = (P(2:n)-P(1:n-1))./(V(2:n)-V(1:n-1));  %Calculate numerical derivative
    
    % Find Pa and Pb for calculation
    if T < Tc
        

        % Break up the data into three types based on spinodal region
        exitflag = 0;
        index = 1;
        while exitflag == 0;
            if dP_dV_data(index) > 0 
                index_Pa = index;
                Va(j) = V(index);
                Pa(j) = P(index);
                exitflag = 1;
            end
            index = index + 1;
        end
        exitflag = 0;
        while exitflag == 0;
            if dP_dV_data(index) < 0 
                index_Pb = index;
                Vb(j) = V(index);
                Pb(j) = P(index);
                exitflag = 1;
            end
            index = index + 1;
        end
        
        
        % Extract co-existence in the function (below) called fugacity_min
        
        [P_vap(j),VL(j),VV(j)] = fugacity_min(Pb(j),a,b,alpha,R,T);
        
        
     % Make Pretty Plots that can selected or removed in postprocessing.
     % Stability data
                
        V1 = V(1:index_Pa);
        V2 = V(index_Pa:index_Pb);
        V3 = V(index_Pb:n);
        P1 = P(1:index_Pa);
        P2 = P(index_Pa:index_Pb);
        P3 = P(index_Pb:n);
        
        for i = 1:n
            if VL(j) < V(i) && VV(j) > V(i)
                P(i) = P_vap(j);
            end
        end
        
        figure(1)
        semilogx(V1,P1,'LineWidth',2,'Color',[1 1 0])
        semilogx(V3,P3,'LineWidth',2,'Color',[1 1 0])
        axis([0.01,10,-5,10])
        title('Peng Robinson Graph')
        ylabel('Pressure')
        xlabel('Volume')
        hold on
        semilogx(V2,P2,'LineWidth',2,'LineStyle',':',...
            'Color',[0.7 0.7 0.7])
    end

    
        semilogx(V,P,'LineWidth',2)
       
    

    
end
Va = [Va,Vc];
Pa = [Pa,Pc];
Vb = [Vb,Vc];
Pb = [Pb,Pc];
semilogx(Va,Pa,'LineWidth',2,'Color',[1 0 0])
semilogx(Vb,Pb,'LineWidth',2,'Color',[1 0 0])

VL = [VL,Vc];
VV = [VV,Vc];
P_vap = [P_vap,Pc];
semilogx(VL,P_vap,'LineWidth',2,'Color',[1 0 0])
semilogx(VV,P_vap,'LineWidth',2,'Color',[1 0 0])

end  %End of main function




function [P_vap,VL,VV] = fugacity_min(Pb,a,b,alpha,R,T)
   
  error = 1;
  Tol = 1E-3;
  P = Pb;%;/2;   %Initialize initial guess.  Pb is always positive.
              
  while error > Tol
      
    A = a*alpha*P/(R*T)^2;
    B = P*b/(R*T);

    % Cubic equation of state:  Z^3 + p1*Z^2 + p2*Z + p3 = 0
    p1 = -1 + B;
    p2 = A - 3*B^2 - 2*B;
    p3 = -A*B + B^2 + B^3;

    coef = [1 p1 p2 p3];
    Z_out = roots(coef);
    ZV = max(Z_out);
    ZL = min(Z_out);
    ln_fv = (ZV-1) - log(ZV-B) - A/(2*sqrt(2)*B)*log((ZV+(1+sqrt(2))*B)/((ZV+(1-sqrt(2))*B)));
    ln_fl = (ZL-1) - log(ZL-B) - A/(2*sqrt(2)*B)*log((ZL+(1+sqrt(2))*B)/((ZL+(1-sqrt(2))*B)));
    
    error = abs(exp(ln_fl)/exp(ln_fv)-1) ;
    P = P*(exp(ln_fl)/exp(ln_fv));
  end
VL = R*T*ZL/P;
VV = R*T*ZV/P;
P_vap =P;
end

function P = PR_EOS(a,b,alpha,R,T,V)

P = R*T/(V-b)-a*alpha/(V^2+2*b*V-b^2);

end

