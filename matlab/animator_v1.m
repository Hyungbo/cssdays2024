%
% Animation of layered coordination of heterogeneous agents
%
% Hyungbo Shim and Hyeonyeong Jang 
% Sept 23, 2024
%
% version 1.0
%
%% Usage:
% space = toggle coupling on / off
% q = quit
%%
% Behavior changes every time because of random generations
%

clear all; clc; close all

N = 11;     % Numbers of agents, must be odd number
AL = 10;    % Arena Length: AL x AL m
MD = 2;     % Minimum distance from other agents
CR = 5;     % Connecting range: within this range, two agents are connected

Ani_Skip = 30;            % Animation skip count for speed 
PT = 0.02;  % pause time for animation
SP = 200;   % Sun lighting period
DF = false; % display progress time
MSF = false;    % Movie save flag
MFN = 'test';   % avi file name

global SF QF
SF = false; % Synchronization flag; alternating whenever 'space' pressed
QF = false; % If true, quit simulation

%%% Random position generation
flag = true;
while flag
    positions = AL*rand(2,N);   % random positions in 10m x 10m ground
    [D,L] = pdist2(positions',CR);   % Get distances of all pairs, and within 5m, connect!
    if all(D > 2)  % minimum distance is 2m; if not, re-sample
        flag = false;
    end
end
% L is now the Laplacian matrix

% Define agents
Agents = [];
for i=1:N
    pp = 0.1;   % perturbation percentage, e.g. 0.1 = 10%
    if (rand < 0.5)     % type 1 generation probability 0.5
        % Type 1 agent with small perturbation
        mu = 0.01;      % related to shape
        nu = 1.2 * (1 + (-pp + 2*pp*rand));       % related to frequency
    else 
        % Type 2 agent with small perturbation
        mu = 2; 
        nu = 1 * (1 + (-pp + 2*pp*rand));
    end
    temp = Agent;   % Agent.m contains the info
    temp.position = positions(:,i);
    temp.mu = mu;
    temp.nu = nu;
    temp.dynamics = @(x,u2,u3,sun) [-x(1)+x(2); ...
                (1-mu*(x(1)^2-1))*(-x(1)+x(2)) - nu*x(1) + u2; ...
                sign(sun-x(3)) + u3];   % x1,x2 : van der pol, x3: median solver
    temp.state = [-1 + 2*rand(2,1); rand]; % initial conditions
    Agents = [Agents; temp];
end

itheta = @(x,mu) (4-3*tanh(tanh(mu-1.025)*5*(x-0.5))); % inverse of theta function

%%% Simulation
hFig = figure;
set(hFig,'WindowKeyPressFcn',@keyPressCallback);
hold on
axis([-1,AL+1,-1,AL+1])   % Area setting with margin 1m

dt = 0.01;          % Time step for integration using RK4 (local function)

% graph handles, to be used for deletion of the graph objects
H1 = zeros(1,N);
H2 = zeros(1,N);
Hg1 = 0;
Hg2 = 0;

disp('Press [q] to quit.')
disp('Press [space] to toggle coupling.')

if MSF  % Movie save flag
    v = VideoWriter(MFN); % 비디오 파일 이름 설정
    v.FrameRate = 1/(dt*Ani_Skip); % 프레임 속도 설정 (초당 프레임 수)
    open(v);
    disp('A movie file is being created in the same directory.')
end

% simulation and animation loop
time = 0;
Ani_flag = true;
Ani_count = 0;
while ~QF
    if Ani_count > Ani_Skip
        Ani_flag = true;
    end
    lw = min( mod(time,SP), SP - mod(time,SP) );   % lightwidth
    lp = 2*lw/SP*AL; % lightposition

    if Ani_flag
        if DF
            disp(['Simulation time = ',num2str(time)])    % for display progress
        end
        % Draw the ground and the sun light 
        Xground = [-1, AL+1, AL+1, -1];
        Yground = [-1, -1, AL+1, AL+1];
        Hg1 = patch(Xground,Yground,'k', 'FaceColor','white','Edgecolor','none');
        Xground = [-1, lp, lp, -1];
        Yground = [-1, -1, AL+1, AL+1];
        Hg2 = patch(Xground,Yground,'k', 'FaceColor','yellow','Edgecolor','none');
    end

    for i = 1:N
        if Ani_flag
            % Draw nodes and breath
            w = 0.2;    % box width/2
            Xnodebox = [positions(1,i)-w, positions(1,i)+w, positions(1,i)+w, positions(1,i)-w];
            Ynodebox = [positions(2,i)-3*w, positions(2,i)-3*w, positions(2,i)+3*w, positions(2,i)+3*w];
            h = Agents(i).state(2);     % y-axis value of Oscillator
            h = abs(h/5*(3*w));         % normalization; '5' is approx maximum of x2 of van der pol Type 2
            % draw node
            if SF
                linec = 0.8*[1,1,1];    % gray colors
            else
                linec = 'black';
            end
            H1(i) = patch(Xnodebox, Ynodebox, 'k', 'EdgeColor', linec, 'FaceColor', 'white', 'LineWidth', 2);  % node frame
            Ynodebox = [positions(2,i)-h, positions(2,i)-h, positions(2,i)+h, positions(2,i)+h];
            color = [0.4, Agents(i).mu/2.5+(Agents(i).nu-0.9), 1-Agents(i).mu/2.5-(Agents(i).nu-0.9)];
            % draw breath
            H2(i) = patch(Xnodebox, Ynodebox, color, 'FaceColor', color, 'LineWidth', 1, 'EdgeColor', 'none'); % node content
        end

        if SF   % Synch Flag
            u2 = 0;
            u3 = 0;
            for j = 1:N
                if L(i,j) < 0
                    u2 = u2 + (Agents(j).state(2) - Agents(i).state(2));
                    u3 = u3 + (Agents(j).state(3) - Agents(i).state(3));
                end
            end
            k2 = 1; % coupling gain
            k3 = 5;
            u2 = k2*u2*itheta(Agents(i).state(3),Agents(i).mu);
            u3 = k3*u3;            
        else
            u2 = 0;
            u3 = 0;
        end

        % Progress
        Agents(i).state = RK4(dt, Agents(i).dynamics, Agents(i).state, u2, u3, Agents(i).position(1) < lp);
    end

    if Ani_flag
        drawnow
        if MSF
            writeVideo(v, getframe); % 각 프레임을 비디오 파일에 기록
        end
        pause(PT)
        delete(Hg1)
        delete(Hg2)
        for i = 1:N
            delete(H1(i))
            delete(H2(i))
        end
        Ani_flag = false;
        Ani_count = 0;
    end

    time = time + dt;
    Ani_count = Ani_count + 1;
end

if MSF
    close(v);
    disp('Movie has been saved');
end
close all;


%%
function x_next = RK4(dt,f,x,u2,u3,sun)
    k1 = f(x, u2, u3, sun);
    k2 = f(x + 0.5*dt*k1, u2, u3, sun);
    k3 = f(x + 0.5*dt*k2, u2, u3, sun);
    k4 = f(x + dt*k3, u2, u3, sun);
    
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
end


function [D,L] = pdist2(X,R)
    % X is a matrix where each row is an observation
    % D will be the vector of pairwise distances between rows
    % R: radius to connect
    
    [n, ~] = size(X);      % Get the number of observations
    D = zeros(n*(n-1)/2, 1); % Preallocate memory for distances
    k = 1;  % Index for the distance vector
    L = zeros(n,n);

    for i = 1:n-1
        for j = i+1:n
            % Compute Euclidean distance between row i and row j
            diff = X(i, :) - X(j, :);
            D(k) = sqrt(sum(diff .^ 2));
            if (D(k) < R)
                L(i,j) = -1;
            end
            k = k + 1;
        end
    end
    L = L + L';
    dia = -sum(L);
    L = L + diag(dia);
end


function keyPressCallback(source,eventdata)

global SF QF

    % determine the key that was pressed
    keyPressed = eventdata.Key;
    if strcmpi(keyPressed,'space')
        SF = ~SF;
    end
    if strcmpi(keyPressed,'q')
        QF = true;
    end
end 

