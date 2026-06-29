cask "wigglegram-creator" do
  version "0.1.2"
  sha256 "cf463e1e451a2e37c11d07829465315168169db3511ca541e6b1005e3a79e6c2"

  url "https://github.com/wjhrdy/wigglegram_creator/releases/download/v#{version}/wigglegram-creator-macOS-arm64.zip"
  name "Wigglegram Creator"
  desc "Create animated wigglegram GIFs and looping videos from image sequences"
  homepage "https://github.com/wjhrdy/wigglegram_creator"

  livecheck do
    url :url
    strategy :github_latest
  end

  # Only an Apple Silicon build is published; Intel Macs can run it under Rosetta.
  depends_on arch:  :arm64
  depends_on macos: :monterey

  app "Wigglegram Creator.app"

  zap trash: "~/Library/Saved Application State/Wigglegram Creator.savedState"

  caveats <<~EOS
    Wigglegram Creator is not signed or notarized, so macOS Gatekeeper will
    block it on first launch. Either right-click the app and choose Open, or
    clear the quarantine flag:

      xattr -dr com.apple.quarantine "#{appdir}/Wigglegram Creator.app"
  EOS
end
