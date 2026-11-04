function R = magnet_fd3d(geo, h)
% magnet_fd3d - 3D finite-difference magnetostatics of the NdFeB H-frame magnet.
%
%   R = magnet_fd3d(geo, h)
%
% Solves div( mu_r grad u ) = div( Br ) for the reduced potential u = mu0*phi
% (so B = -mu_r grad u + Br, in tesla) on a uniform cell-centred grid of
% spacing h (mm), using 1/8 symmetry:
%   x = 0 : u = 0 (antisymmetric; B0 is along x and Br is even in x)
%   y = 0, z = 0 : zero normal flux (mirror planes)
%   outer box : u = 0
% Materials are linear: air (1), NdFeB (geo.mur_mag, remanence geo.Br along +x),
% steel (geo.mur_fe). The sample-access hole in the top bar is not modelled.
%
% geo fields (mm unless noted):
%   g      gap between pole-cap faces          Rm   magnet / cap radius
%   tm     magnet thickness                    tc   pole-cap thickness (0 = none)
%   tp     side-plate thickness                Hy   half height of side plates
%   Dy     half depth of yoke                  tb   top/bottom bar thickness
%   yoke   true/false                          pad  air margin around the steel
%   Br     remanence, T                        mur_mag, mur_fe
%   rimh, rimw  height and radial width of a raised rim (Rose shim) on each pole cap
%   rs, zs sample radius and half-length (tube axis along z)
%
% Returns a struct with the grid, |B| on cell centres, B at the centre,
% homogeneity over the sample volume and peak flux density in the steel.

if nargin < 2, h = 1; end
d = struct('g',20,'Rm',25,'tm',15,'tc',5,'tp',20,'Hy',70,'Dy',30,'tb',20, ...
           'yoke',true,'pad',40,'Br',1.3,'mur_mag',1.05,'mur_fe',1000, ...
           'rs',5,'zs',5,'slab',false,'rimh',0,'rimw',0);
fn = fieldnames(d);
for k = 1:numel(fn), if ~isfield(geo,fn{k}), geo.(fn{k}) = d.(fn{k}); end, end

xc = geo.g/2; xm = xc + geo.tc; xp = xm + geo.tm; xo = xp + geo.tp;
if geo.yoke
    Lx = xo + geo.pad; Ly = geo.Dy + geo.pad; Lz = geo.Hy + geo.tb + geo.pad;
else
    Lx = xp + 60; Ly = geo.Rm + 60; Lz = geo.Rm + 60;
end
nx = round(Lx/h); ny = round(Ly/h); nz = round(Lz/h);
x = ((1:nx)'-0.5)*h; y = ((1:ny)'-0.5)*h; z = ((1:nz)'-0.5)*h;
[X,Y,Z] = ndgrid(x,y,z);
r = hypot(Y,Z);
if geo.slab, r = 0*r; Y = 0*Y; Z = 0*Z; end   % 1D validation: infinite slabs

% ---- materials ----
mu = ones(nx,ny,nz); Brx = zeros(nx,ny,nz);
mag = (X >= xm) & (X < xp) & (r <= geo.Rm);
mu(mag) = geo.mur_mag; Brx(mag) = geo.Br;
cap = (X >= xc) & (X < xm) & (r <= geo.Rm);
if geo.rimh > 0   % Rose-type shim: raised steel rim on the pole-cap face
    cap = cap | ((X >= xc - geo.rimh) & (X < xc) & (r <= geo.Rm) & (r > geo.Rm - geo.rimw));
end
steel = cap;
if geo.yoke
    plate = (X >= xp) & (X < xo) & (Y <= geo.Dy) & (Z <= geo.Hy);
    bar   = (X < xo) & (Y <= geo.Dy) & (Z > geo.Hy) & (Z <= geo.Hy + geo.tb);
    steel = steel | plate | bar;
end
mu(steel) = geo.mur_fe;

% ---- assemble  sum_f mu_f (u_c - u_n) = rhs ----
N = nx*ny*nz; id = reshape(1:N, nx, ny, nz);
hm = @(a,b) 2*a.*b./(a+b);
I = []; J = []; V = []; diagA = zeros(N,1); b = zeros(N,1);

% x faces (interior)
mf = hm(mu(1:end-1,:,:), mu(2:end,:,:));
% remanent flux through a face: two half-cells in series ->
%   F = -mu_f du/h + mu_f*(Br_L/(2 mu_L) + Br_R/(2 mu_R)),  mu_f = harmonic mean
bf = mf .* (Brx(1:end-1,:,:)./(2*mu(1:end-1,:,:)) + Brx(2:end,:,:)./(2*mu(2:end,:,:)));
cL = id(1:end-1,:,:); cR = id(2:end,:,:);
add_pairs(cL(:), cR(:), mf(:));
b = b + accumarray(cL(:), -h*bf(:), [N 1]) + accumarray(cR(:), h*bf(:), [N 1]);
% x = 0 plane: u = 0 at the face (antisymmetry)
c0 = id(1,:,:); diagA(c0(:)) = diagA(c0(:)) + 2*reshape(mu(1,:,:),[],1);
b(c0(:)) = b(c0(:)) + h*reshape(Brx(1,:,:),[],1);
% outer x face: u = 0
c1 = id(end,:,:); diagA(c1(:)) = diagA(c1(:)) + 2*reshape(mu(end,:,:),[],1);

% y faces (interior); y = 0 is a mirror (no term); outer y: u = 0
mf = hm(mu(:,1:end-1,:), mu(:,2:end,:));
cL = id(:,1:end-1,:); cR = id(:,2:end,:); add_pairs(cL(:), cR(:), mf(:));
c1 = id(:,end,:); if ~geo.slab, diagA(c1(:)) = diagA(c1(:)) + 2*reshape(mu(:,end,:),[],1); end

% z faces (interior); z = 0 mirror; outer z: u = 0
mf = hm(mu(:,:,1:end-1), mu(:,:,2:end));
cL = id(:,:,1:end-1); cR = id(:,:,2:end); add_pairs(cL(:), cR(:), mf(:));
c1 = id(:,:,end); if ~geo.slab, diagA(c1(:)) = diagA(c1(:)) + 2*reshape(mu(:,:,end),[],1); end

A = sparse([I; (1:N)'], [J; (1:N)'], [V; diagA], N, N);

% ---- solve (Jacobi-scaled PCG with incomplete Cholesky) ----
s = 1./sqrt(full(diag(A))); S = spdiags(s,0,N,N);
As = S*A*S; bs = s.*b;
try
    L = ichol(As, struct('type','ict','droptol',1e-3,'michol','on'));
catch
    L = ichol(As, struct('type','ict','droptol',1e-3,'diagcomp',1e-3));
end
[us, flag, relres, iter] = pcg(As, bs, 1e-9, 5000, L, L');
u = reshape(s.*us, nx, ny, nz);
if flag ~= 0, warning('pcg flag %d, relres %.1e', flag, relres); end

% ---- B on cell centres (average of the two faces in each direction) ----
upad = cat(1, -u(1,:,:), u, -u(end,:,:));        % ghost cells: antisym at x=0, u=0 outside
mpad = cat(1, mu(1,:,:), mu, mu(end,:,:));
bpad = cat(1, Brx(1,:,:), Brx, Brx(end,:,:));     % Br is even in x (mirror ghost at x=0)
bpad(end,:,:) = 0;
mfx  = hm2(mpad);
Fx = -mfx .* diff(upad,1,1)/h + mfx .* (bpad(1:end-1,:,:)./(2*mpad(1:end-1,:,:)) + bpad(2:end,:,:)./(2*mpad(2:end,:,:)));
Bx = 0.5*(Fx(1:end-1,:,:) + Fx(2:end,:,:));
upad = cat(2, u(:,1,:), u, -u(:,end,:));          % mirror at y=0, u=0 outside
if geo.slab, upad(:,end,:) = u(:,end,:); end
Fy = -hm(cat(2, mu(:,1,:), mu(:,1:end-1,:)), mu) .* diff(upad(:,1:end-1,:),1,2)/h;   % lower faces
Fy2 = -hm(mu, cat(2, mu(:,2:end,:), mu(:,end,:))) .* diff(upad(:,2:end,:),1,2)/h;   % upper faces
By = 0.5*(Fy + Fy2);
upad = cat(3, u(:,:,1), u, -u(:,:,end));
if geo.slab, upad(:,:,end) = u(:,:,end); end
Fz = -hm(cat(3, mu(:,:,1), mu(:,:,1:end-1)), mu) .* diff(upad(:,:,1:end-1),1,3)/h;
Fz2 = -hm(mu, cat(3, mu(:,:,2:end), mu(:,:,end))) .* diff(upad(:,:,2:end),1,3)/h;
Bz = 0.5*(Fz + Fz2);
Bmag = sqrt(Bx.^2 + By.^2 + Bz.^2);

% ---- results ----
R.geo = geo; R.h = h; R.x = x; R.y = y; R.z = z;
R.Bx = Bx; R.Bmag = Bmag; R.mu = mu; R.iter = iter; R.relres = relres;
R.B0 = Bx(1,1,1);
smp = (hypot(X,Y) <= geo.rs) & (Z <= geo.zs);    % sample: tube radius rs, |z| <= zs
Bs = Bx(smp);
R.sample_ppm_pp  = (max(Bs) - min(Bs))/R.B0*1e6;
R.sample_ppm_std = std(Bs)/R.B0*1e6;
R.Bsteel_max = max(Bmag(steel & ~cap));
R.Bsteel_p99 = prctile(Bmag(steel & ~cap), 99);

    function add_pairs(a, c, w)
        I = [I; a; c]; J = [J; c; a]; V = [V; -w; -w];
        diagA = diagA + accumarray(a, w, [N 1]) + accumarray(c, w, [N 1]);
    end
    function m = hm2(mm)
        m = 2*mm(1:end-1,:,:).*mm(2:end,:,:)./(mm(1:end-1,:,:)+mm(2:end,:,:));
    end
end
