cask "wigglegram-creator" do
  version "0.5.0"
  sha256 "20465f72b72c12980c8e118477178a5904f919b1e6ce8571ef626decaa12052c"

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
