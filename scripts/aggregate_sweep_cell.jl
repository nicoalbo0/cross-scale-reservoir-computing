# Aggregate a single sweep cell across seeds.
# Reads enso_temporal_field_preds_<mode>_seed*.jld2 in <celldir>; writes a one-line
# CSV row to <out_csv> with the cell's id (passed in), per-seed values, and
# mean ± std for: acc12, rmse12, std_ratio12, pc3, pc12, ppacc_n34_mean,
# ppacc_global_mean. Also computes a composite score:
#   composite = 0.4·ACC_12 + 0.3·pc_3 + 0.3·(1 − |1 − std_ratio_12|)
#
# Usage: julia --project=. scripts/aggregate_sweep_cell.jl <celldir> <out_csv> <cell_id>
#
# If <out_csv> doesn't exist, a header row is written first.
# Skips cells with zero per-seed JLD2 files.

using JLD2, Statistics, Glob

length(ARGS) ≥ 3 || error("Usage: aggregate_sweep_cell.jl <celldir> <out_csv> <cell_id>")
celldir, out_csv, cell_id = ARGS

paths = sort(glob("enso_temporal_field_preds_*_seed*.jld2", celldir))
if isempty(paths)
    @info "no per-seed files in $celldir; skipping"
    exit(0)
end

acc12s = Float64[]; rmse12s = Float64[]; sr12s = Float64[]
pc3s = Float64[]; pc12s = Float64[]
ppn34s = Float64[]; ppglob = Float64[]
seeds = Int[]

for p in paths
    jldopen(p, "r") do f
        push!(seeds,    Int(f["seed"]))
        push!(acc12s,   Float64(f["acc12"]))
        push!(rmse12s,  Float64(f["rmse12"]))
        push!(sr12s,    Float64(f["std_ratio12"]))
        push!(pc3s,     Float64(f["pc3"]))
        push!(pc12s,    Float64(f["pc12"]))
        push!(ppn34s,   Float64(f["ppacc_n34_mean"]))
        push!(ppglob,   Float64(f["ppacc_global_mean"]))
    end
end

m(x) = isempty(x) ? NaN : mean(x)
s(x) = length(x) < 2 ? NaN : std(x)

composite = 0.4 * m(acc12s) + 0.3 * m(pc3s) + 0.3 * (1 - abs(1 - m(sr12s)))

header = "cell_id,n_seeds,seeds," *
         "acc12_mean,acc12_std," *
         "rmse12_mean,rmse12_std," *
         "std_ratio12_mean,std_ratio12_std," *
         "pc3_mean,pc3_std," *
         "pc12_mean,pc12_std," *
         "ppacc_n34_mean_mean,ppacc_n34_mean_std," *
         "ppacc_global_mean_mean,ppacc_global_mean_std," *
         "composite\n"

write_header = !isfile(out_csv) || filesize(out_csv) == 0
open(out_csv, "a") do io
    write_header && write(io, header)
    seed_str = join(seeds, ";")
    row = "$(cell_id),$(length(seeds)),$(seed_str)," *
          "$(m(acc12s)),$(s(acc12s))," *
          "$(m(rmse12s)),$(s(rmse12s))," *
          "$(m(sr12s)),$(s(sr12s))," *
          "$(m(pc3s)),$(s(pc3s))," *
          "$(m(pc12s)),$(s(pc12s))," *
          "$(m(ppn34s)),$(s(ppn34s))," *
          "$(m(ppglob)),$(s(ppglob))," *
          "$(composite)\n"
    write(io, row)
end
println("✓ $(cell_id): n=$(length(seeds))  acc12=$(round(m(acc12s);digits=3))±$(round(s(acc12s);digits=3))  pc3=$(round(m(pc3s);digits=3))  sr12=$(round(m(sr12s);digits=3))  composite=$(round(composite;digits=3))")
