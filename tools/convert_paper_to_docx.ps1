param(
    [string]$InputPath = "PAPER_DRAFT.md",
    [string]$OutputPath = "Blood_Cell_Analysis_Paper_Draft.docx"
)

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

function Escape-Xml([string]$Text) {
    return [System.Security.SecurityElement]::Escape($Text)
}

function Paragraph([string]$Text, [string]$Style = "Normal") {
    $safe = Escape-Xml $Text
    return "<w:p><w:pPr><w:pStyle w:val=`"$Style`"/></w:pPr><w:r><w:t xml:space=`"preserve`">$safe</w:t></w:r></w:p>"
}

if (-not (Test-Path -LiteralPath $InputPath)) {
    throw "Input file not found: $InputPath"
}

$body = New-Object System.Text.StringBuilder
$inCode = $false
foreach ($line in Get-Content -LiteralPath $InputPath) {
    if ($line -match '^```') { $inCode = -not $inCode; continue }
    if ($inCode) { [void]$body.Append((Paragraph $line "Code")); continue }
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line -match '^# (.+)$')       { [void]$body.Append((Paragraph $Matches[1] "Title")); continue }
    if ($line -match '^## (.+)$')      { [void]$body.Append((Paragraph $Matches[1] "Heading1")); continue }
    if ($line -match '^### (.+)$')     { [void]$body.Append((Paragraph $Matches[1] "Heading2")); continue }
    if ($line -match '^[-*] (.+)$')    { [void]$body.Append((Paragraph ("• " + $Matches[1]) "Normal")); continue }
    if ($line -match '^\|') {
        $cells = $line.Trim('|').Split('|') | ForEach-Object { $_.Trim() }
        if ($cells -notmatch '^[-: ]+$') { [void]$body.Append((Paragraph ($cells -join "`t") "Table")) }
        continue
    }
    # Keep equations and Markdown emphasis readable in Word; remove only markup.
    $text = $line -replace '\*\*', '' -replace '`', ''
    [void]$body.Append((Paragraph $text "Normal"))
}

$documentXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>$($body.ToString())<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/></w:sectPr></w:body></w:document>
"@

$stylesXml = @"
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:sz w:val="22"/><w:szCs w:val="22"/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:pPr><w:jc w:val="center"/><w:spacing w:after="240"/></w:pPr><w:rPr><w:b/><w:sz w:val="32"/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="Heading 1"/><w:pPr><w:spacing w:before="240" w:after="120"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="Heading 2"/><w:pPr><w:spacing w:before="160" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="23"/><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Table"><w:name w:val="Table Text"/><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="18"/></w:rPr></w:style>
</w:styles>
"@

$contentTypes = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/></Types>'
$rootRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
$docRels = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'

$fullOutput = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $fullOutput) { Remove-Item -LiteralPath $fullOutput -Force }
$archive = [System.IO.Compression.ZipFile]::Open($fullOutput, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    $entries = @{
        '[Content_Types].xml' = $contentTypes
        '_rels/.rels' = $rootRels
        'word/document.xml' = $documentXml
        'word/styles.xml' = $stylesXml
        'word/_rels/document.xml.rels' = $docRels
    }
    foreach ($name in $entries.Keys) {
        $entry = $archive.CreateEntry($name)
        $writer = New-Object System.IO.StreamWriter($entry.Open(), [System.Text.Encoding]::UTF8)
        try { $writer.Write($entries[$name]) } finally { $writer.Dispose() }
    }
} finally { $archive.Dispose() }

Write-Output "Created: $fullOutput"
