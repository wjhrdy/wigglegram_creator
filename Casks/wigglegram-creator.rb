cask "wigglegram-creator" do
  version "0.1.1"
  sha256 "e39bdf7af8692a52cf4c7c44da9b51ebfcb2fcfa55b13de37a9262398648a5f2"

  url "https://github.com/wjhrdy/wigglegram_creator/releases/download/v#{version}/wigglegram-creator-macOS-arm64.zip",
      verified: "github.com/wjhrdy/wigglegram_creator/"
  name "Wigglegram Creator"
  desc "Create animated wigglegram GIFs and looping videos from image sequences"
  homepage "https://github.com/wjhrdy/wigglegram_creator"

  livecheck do
    url :url
    strategy :github_latest
  end

  # Only an Apple Silicon build is published; Intel Macs can run it under Rosetta.
  depends_on arch: :arm64

  app "Wigglegram Creator.app"

  zap trash: [
    "~/Library/Saved Application State/Wigglegram Creator.savedState",
  ]

  caveats <<~EOS
    Wigglegram Creator is not signed or notarized, so macOS Gatekeeper will
    block it on first launch. If you installed without --no-quarantine, either
    right-click the app and choose Open, or run:

      xattr -dr com.apple.quarantine "#{appdir}/Wigglegram Creator.app"
  EOS
end
