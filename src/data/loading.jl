"""
    load_data(Q, L, μ; show_data=false, refinement)

Load Kuramoto–Sivashinsky (KS) data. Dispatches to `load_kuramoto_data`.
"""
load_data(Q::Int, L::Int, μ::Real; show_data::Bool=false, refinement::Int) = load_kuramoto_data(Q, L, μ; show_data=show_data, refinement=refinement)

"""
    load_data(res; show_data=false, refinement)

Load SST (sea surface temperature) data for given resolution(s). Dispatches to `load_sst_data`.
"""
load_data(res::Vector{Float64}; show_data::Bool=false, refinement::Int) = load_sst_data(res; show_data=show_data, refinement=refinement)

"""
    load_kuramoto_data(Q, L, μ; show_data=false, refinement)

Load Kuramoto–Sivashinsky data from CSV. Data is normalized (zero mean, unit variance).
Optionally interpolated in time by `refinement`. Returns `(data, dt)` where `data` is (L×T).
"""
function load_kuramoto_data(Q::Int, L::Int, μ::T; show_data::Bool=false, refinement::Int) where T<:Real

    filepath = pwd()*"/data/kuramoto/Q$(Q)_L$(L)_mu$(μ)_ks_data.csv"
    dt = 0.25

    f = Matrix(CSV.read("$filepath", DataFrame))';
    data = (f .- mean(f,dims=2)) ./ std(f, dims=2)

    if refinement > 1
        data = cubic_time_interpolate(data, dt, refinement)
    end
        
    if show_data
        p = heatmap(data[:, 1:5000]; title="KS data sample", xlabel="t", ylabel="row (state)")
        display(p)
    end
    
    return data, dt / refinement

end

"""
    load_sst_data(resolutions_vec; show_data=false, refinement)

Load SST data for each resolution in `resolutions_vec`. Uses cleaned JLD2 files
(`sst_clean_<res>.jld2`); creates them from `sst_final_<res>.jld2` if needed.
Temporal/spatial cleaning and normalization applied. Returns `(data_vec, dt)` where
`data_vec` is a vector of (nlon×nlat×nt) arrays.
"""
function load_sst_data(resolutions_vec::Vector{T}; show_data::Bool=false, refinement::Int) where T<:Real

    dir = pwd()*"/data/sst"

    data_vec = Vector{Array{Float64, 3}}(undef, length(resolutions_vec))

    for (res_i, res) in enumerate(resolutions_vec)

        if !isfile(dir*"/sst_clean_$(res).jld2")

            f = jldopen(dir*"/sst_final_$(res).jld2")
            data = f["sst"]
            close(f)

            nlon, nlat, nt = size(data)
            nan_threshold = nt * 0.15

            # temporal cleaning
            for i in axes(data,1), j in axes(data,2)

                row = @view data[i, j, :]

                if count(isnan, row) > nan_threshold
                    row .= NaN
                else
                    times = findall(!isnan, row)

                    if !isempty(times)
                        t = LinRange(0, 1, length(times))
                        vals = row[times]

                        itp = Interpolations.scale(
                            interpolate(hcat(times, vals),
                                (BSpline(Cubic(Natural(OnGrid()))), NoInterp())),
                            t, 1:2
                        )

                        ti = LinRange(0,1,nt)
                        row .= [itp(ti[k], 2) for k in eachindex(ti)]
                    end
                end
            end

            # spatial cleaning ----
            for i in 2:nlon-1, j in 2:nlat-1

                counter = 0
                for ii in i-1:i+1, jj in j-1:j+1
                    if isnan(data[ii, jj, 1])
                        counter += 1
                    end
                end

                if counter >= 7
                    data[i,j,:] .= NaN
                end
            end

            @assert size(data) == (nlon, nlat, nt)

            jldsave(dir*"/sst_clean_$(res).jld2", sst=data)

        else
                
            f = jldopen(dir*"/sst_clean_$(res).jld2")
            data = f["sst"]
            close(f)
        end

        if show_data

            p = heatmap(data[:, :, 1]', title="Sample data", 
            framestyle=:box, colorbar_title="\nCounts",
            xlabel="Longitude", ylabel="Latitude",
            right_margin=5mm)
            display(p)

        end

        if refinement > 1
            nlon, nlat, nt = size(data)
            data = reshape(data, nlon* nlat, nt)

            data = cubic_time_interpolate(data, 1.0, refinement)
            data = reshape(data, nlon, nlat, size(data, 2))
        end

        data = reshape(data, nlon* nlat, size(data,3))
        data = (data .- mean(data, dims=2)) ./ (std(data, dims=2))
        data = reshape(data, nlon, nlat, size(data, 2))
        data[isnan.(data)] .= 0.0

        data_vec[res_i] = data
    end

    return data_vec, 1 / refinement

end