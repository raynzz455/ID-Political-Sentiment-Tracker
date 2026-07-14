# git-commit-scheduler.ps1
 $batchSize = 4       # Jumlah file per commit
 $commitsPerDay = 3   # Jumlah commit per hari (bisa diubah jadi 4)
 $commitCount = 0
 $dayOffset = 0
 $currentDate = Get-Date

# Ambil semua file yang ada perubahan (modified/untracked)
 $files = git status --porcelain | Where-Object { $_ -match "^\s?[M|?]" } | ForEach-Object { $_.Trim().Substring(3) }

if ($files.Count -eq 0) {
    Write-Host "Tidak ada perubahan untuk di-commit." -ForegroundColor Yellow
    exit
}

Write-Host "Ditemukan $($files.Count) file. Memulai proses commit bertahap..." -ForegroundColor Cyan

for ($i = 0; $i -lt $files.Count; $i += $batchSize) {
    # Ambil batch file
    $batch = $files[$i..([Math]::Min($i + $batchSize - 1, $files.Count - 1))]
    
    # Reset staging area
    git reset HEAD --quiet

    # Add file ke staging
    foreach ($file in $batch) {
        git add "`"$file`""
    }

    # Tentukan tanggal commit
    $commitDate = $currentDate.AddDays($dayOffset).ToString("yyyy-MM-ddTHH:mm:ss")
    
    # Buat commit dengan tanggal custom (Author & Committer date diset agar GitHub graph terbaca)
    $env:GIT_COMMITTER_DATE = $commitDate
    $commitMsg = "feat(pipeline): batch update $($commitCount + 1) - refactoring & testing"
    
    git commit -m $commitMsg --date="$commitDate" | Out-Null
    
    Write-Host "  ✅ Commit $($commitCount + 1) dibuat (Tanggal: $($currentDate.AddDays($dayOffset).ToString('yyyy-MM-dd'))) - $($batch.Count) file" -ForegroundColor Green
    
    $commitCount++

    # Tambah hari jika sudah mencapai batas commit per hari
    if ($commitCount % $commitsPerDay -eq 0) {
        $dayOffset++
    }
}

Write-Host "`nSelesai! $commitCount commit berhasil dibuat dalam $dayOffset hari." -ForegroundColor Cyan
Write-Host "Silakan jalankan: git push origin main" -ForegroundColor Yellow