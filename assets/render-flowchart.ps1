$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

# Draw at twice the logical size so the README image stays sharp when scaled.
$bitmap = [System.Drawing.Bitmap]::new(1920, 1260)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.ScaleTransform(2, 2)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$graphics.Clear([System.Drawing.ColorTranslator]::FromHtml('#f6f8fb'))

function Draw-Text {
    param([string]$Text, [float]$X, [float]$Y, [float]$Size = 20,
          [string]$Color = '#334155', [switch]$Bold, [float]$MaxWidth = 900)
    $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    $font = [System.Drawing.Font]::new('Segoe UI', $Size, $style, [System.Drawing.GraphicsUnit]::Pixel)
    $brush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($Color))
    try {
        $measured = $graphics.MeasureString($Text, $font)
        if ($measured.Width -gt $MaxWidth) { throw "Text exceeds available width: $Text" }
        $graphics.DrawString($Text, $font, $brush, $X, $Y)
    } finally {
        $font.Dispose()
        $brush.Dispose()
    }
}

function Draw-Card {
    param([float]$X, [float]$Y, [string]$Number, [string]$Title,
          [string[]]$Lines, [string]$Accent = '#2563eb', [string]$Fill = '#ffffff')
    $brush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($Fill))
    $border = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml('#cbd5e1'), 1.5)
    $accentBrush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($Accent))
    try {
        $graphics.FillRectangle($brush, $X, $Y, 272, 152)
        $graphics.DrawRectangle($border, $X, $Y, 272, 152)
        $graphics.FillRectangle($accentBrush, $X, $Y, 5, 152)
        Draw-Text -Text "$Number. $Title" -X ($X + 17) -Y ($Y + 17) -Size 23 -Color $Accent -Bold -MaxWidth 240
        for ($line = 0; $line -lt $Lines.Count; $line++) {
            Draw-Text -Text $Lines[$line] -X ($X + 17) -Y ($Y + 58 + 26 * $line) -Size 19 -MaxWidth 240
        }
    } finally {
        $brush.Dispose()
        $border.Dispose()
        $accentBrush.Dispose()
    }
}

function Draw-Arrow {
    param([float]$X1, [float]$Y1, [float]$X2, [float]$Y2, [string]$Color = '#64748b')
    $pen = [System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml($Color), 2.5)
    $cap = [System.Drawing.Drawing2D.AdjustableArrowCap]::new(4, 5)
    $pen.CustomEndCap = $cap
    try { $graphics.DrawLine($pen, $X1, $Y1, $X2, $Y2) }
    finally { $pen.Dispose(); $cap.Dispose() }
}

try {
    Draw-Text -Text 'How one knapsack solution is built' -X 30 -Y 25 -Size 30 -Color '#0f172a' -Bold
    Draw-Text -Text 'Each step connects a decision to its effect on capacity and reward.' -X 31 -Y 72 -Size 20

    Draw-Card -X 32 -Y 112 -Number '1' -Title 'New instance' -Lines @('Load item weights,', 'values, and capacity.')
    Draw-Card -X 344 -Y 112 -Number '2' -Title 'Observe state' -Lines @('All item weights / values', 'Current item index', 'Remaining capacity')
    Draw-Card -X 656 -Y 112 -Number '3' -Title 'Choose action' -Lines @('Include if the item fits,', 'or skip the item.')
    Draw-Card -X 656 -Y 366 -Number '4' -Title 'Update' -Lines @('Subtract included weight.', 'Reward: included value,', 'or zero when skipped.')
    Draw-Card -X 344 -Y 366 -Number '5' -Title 'More items?' -Lines @('Yes: advance item index', 'and observe again.', 'No: finish the solution.') -Accent '#a16207' -Fill '#fffbeb'
    Draw-Card -X 32 -Y 366 -Number '6' -Title 'Finish solution' -Lines @('Total reward =', 'total selected value.') -Accent '#047857' -Fill '#ecfdf5'

    Draw-Arrow -X1 307 -Y1 188 -X2 339 -Y2 188
    Draw-Arrow -X1 619 -Y1 188 -X2 651 -Y2 188
    Draw-Arrow -X1 792 -Y1 270 -X2 792 -Y2 359
    Draw-Arrow -X1 650 -Y1 442 -X2 622 -Y2 442
    Draw-Arrow -X1 338 -Y1 442 -X2 310 -Y2 442
    Draw-Arrow -X1 480 -Y1 359 -X2 480 -Y2 271 -Color '#a16207'
    Draw-Text -Text 'Yes' -X 496 -Y 300 -Size 20 -Color '#a16207' -Bold -MaxWidth 140
    Draw-Text -Text 'Next item' -X 496 -Y 327 -Size 18 -Color '#a16207' -MaxWidth 140
    Draw-Text -Text 'No' -X 307 -Y 412 -Size 18 -Color '#64748b' -Bold -MaxWidth 36

    Draw-Text -Text 'Training repeats this process across many instances to improve the policy.' -X 31 -Y 556 -Size 21 -Color '#0f172a'
    Draw-Text -Text 'One episode constructs one solution; this diagram does not show experimental results.' -X 31 -Y 594 -Size 18 -Color '#64748b'
    $outputPath = Join-Path $PSScriptRoot 'rl-knapsack-flow.png'
    $bitmap.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
    Write-Output "Rendered $outputPath (1920 x 1260 pixels). All text fits its allotted width."
} finally {
    $graphics.Dispose()
    $bitmap.Dispose()
}
