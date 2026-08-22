param(
    [string]$Dataset = 'C:\Users\crazy\Documents\Lumen_data\train_ko.jsonl',
    [string]$Model = '',
    [string]$Output = '',
    [int]$MaxSampleTokens = 32,
    [int]$Epochs = 1
)

$repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$trainer = Join-Path $repo 'build\bin\llama-finetune-qlion.exe'
if (-not $Model) {
    $Model = Join-Path $repo 'models\training\Qwen3.5-0.8B-Q4_0-unmixed.gguf'
}
if (-not $Output) {
    $Output = Join-Path $repo 'models\training\Qwen3.5-0.8B-Lumen-ko-QLion-Q4_0.gguf'
}

& $trainer `
    --model $Model `
    --train-file $Dataset `
    --quant-type q4_0 `
    --optimizer qlion `
    --qat-out $Output `
    --qat-max-sample-tokens $MaxSampleTokens `
    -c 256 -b 256 -ub 64 `
    -ngl all -t 4 -tb 4 -dthr 4 `
    -lr 1e-6 -lr-min 1e-7 --lr-scheduler cosine `
    -epochs $Epochs -val-split 0.05 `
    --save-every 25 --shuffle-dataset

exit $LASTEXITCODE
