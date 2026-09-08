
# Usage

This section describes how to use Obliqua. The module can be run three ways:

- **Full spectrum (Standalone)**: Compute the tidal ``k``-Love number response for a full spectrum of forcing frequencies. This mode is agnostic to the orbital parameters that feed into the Hansen mode weights and dissipative response. The generated spectrum can be used as a lookup table, or to probe the quantative dissipative response of the tidal model to a wide range of forcing frequencies.

- **Adaptive (Standalone)**: Compute the tidal ``k``-Love number response for a subset of forcing frequencies that are adaptively selected based on the orbital parameters and Hansen mode weights. This mode targets the physically relevant forcing frequencies and, hence, also allows for the computation of the dissipative response. 

- **Adaptive (PROTEUS)**: By extension of the previous mode, the adaptive mode can be used in conjunction with the PROTEUS framework. This allows Obliqua to interact with both dynamically evolving orbital parameters and interior properties. The tidal ``k``-Love number response is computed on-the-fly and is used to update the orbital evolution while the dissipative response feedsback into the interior. 

Naturally, one can also use the full spectrum mode in conjunction with PROTEUS in post-processing. This can forexample be used to validate the adaptive mode or to study the impact of different forcing frequencies on the tidal response. Moreover, it may be used to test model convergence. Below, we provide here an example run of the full spectrum mode in conjunction with PROTEUS computed in post-processing.

In all cases you configure the model through a configuration file, described in the [configuration guide](@ref "Configuration file"). If you run into problems, see the [troubleshooting](@ref "Troubleshooting") page.

!!! info
    For a quick start, jump to the [tutorials](@ref "Zero-dimensional test case: The Moon") section!

### Example output

The evolution of the tidal heating profile and the tidal ``k``-Love number response is shown below for a 1D poroviscoelastic model with an evolving interior. The left panel shows the tidal heating profile as a function of radius and time, whilst the right panel shows the tidal ``k``-Love number response as a function of forcing frequency and time. The planet is initially fully molten, hence fluid tides dominate the tidal response. As the planet cools, the interior solidifies and the tidal response transitions to a solid dominated regime.

```@raw html
<div align="center" style="max-width: 1100px; margin: 0 auto; font-family: system-ui, -apple-system, sans-serif;">
    <!-- Video Side-by-Side Equal Height Container -->
    <div style="display: flex; gap: 12px; justify-content: center; align-items: stretch; flex-wrap: wrap;">
        <div style="flex: 1 1 0; min-width: 280px; max-width: 530px; display: flex;">
            <video id="vid1" playsinline style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px; display: block;">
                <source src="../panels/heating_evolution_m2_dark.webm" type="video/webm">
            </video>
        </div>
        <div style="flex: 1 1 0; min-width: 280px; max-width: 530px; display: flex;">
            <video id="vid2" playsinline style="width: 100%; height: 100%; object-fit: cover; border-radius: 6px; display: block;">
                <source src="../panels/love_evolution_dark.webm" type="video/webm">
            </video>
        </div>
    </div>

    <!-- Master Controller Bar -->
    <div style="background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 8px; padding: 10px 16px; margin-top: 10px; display: flex; align-items: center; gap: 12px; max-width: 1070px; flex-wrap: wrap;">
        <!-- Play / Pause Button -->
        <button id="master-play" style="background: #0066cc; color: white; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-weight: 600; min-width: 65px;">Play</button>

        <!-- Timeline Slider -->
        <input type="range" id="master-scrubber" min="0" max="100" value="0" step="0.01" style="flex: 1; min-width: 150px; cursor: pointer;">

        <!-- Timestamp Display -->
        <span id="master-time" style="font-family: monospace; font-size: 13px; color: #ccc; white-space: nowrap;">0.00s / 0.00s</span>

        <!-- Direct Time Jump Input -->
        <div style="display: flex; align-items: center; gap: 4px; white-space: nowrap;">
            <input type="number" id="master-input" placeholder="0.0" step="0.1" min="0" style="width: 60px; padding: 4px 6px; border-radius: 4px; border: 1px solid #444; background: #222; color: #fff; font-family: monospace; text-align: center;">
            <button id="master-jump" style="background: #333; color: #eee; border: 1px solid #555; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 12px;">Go</button>
        </div>

        <!-- Playback Speed Control -->
        <div style="display: flex; align-items: center; gap: 4px; white-space: nowrap;">
            <select id="master-speed" style="background: #222; color: #fff; border: 1px solid #444; padding: 4px 6px; border-radius: 4px; font-size: 12px; cursor: pointer; font-family: monospace;">
                <option value="0.25">0.25x</option>
                <option value="0.5">0.5x</option>
                <option value="1" selected>1.0x</option>
                <option value="1.5">1.5x</option>
                <option value="2">2.0x</option>
            </select>
        </div>
    </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function () {
    const v1 = document.getElementById("vid1");
    const v2 = document.getElementById("vid2");
    const playBtn = document.getElementById("master-play");
    const scrubber = document.getElementById("master-scrubber");
    const timeDisplay = document.getElementById("master-time");
    const timeInput = document.getElementById("master-input");
    const jumpBtn = document.getElementById("master-jump");
    const speedSelect = document.getElementById("master-speed");

    function getDuration() {
        return Math.max(v1.duration || 0, v2.duration || 0);
    }

    function formatTime(sec) {
        if (isNaN(sec)) return "0.00s";
        return sec.toFixed(2) + "s";
    }

    function syncTimeTo(seconds) {
        const dur = getDuration();
        const clamped = Math.max(0, Math.min(seconds, dur));
        v1.currentTime = clamped;
        v2.currentTime = clamped;
        scrubber.value = dur ? (clamped / dur) * 100 : 0;
        timeDisplay.textContent = `${formatTime(clamped)} / ${formatTime(dur)}`;
    }

    // Play / Pause Toggle
    playBtn.addEventListener("click", () => {
        if (v1.paused) {
            v1.play();
            v2.play();
            playBtn.textContent = "Pause";
        } else {
            v1.pause();
            v2.pause();
            playBtn.textContent = "Play";
        }
    });

    // Speed Control
    speedSelect.addEventListener("change", () => {
        const speed = parseFloat(speedSelect.value);
        v1.playbackRate = speed;
        v2.playbackRate = speed;
    });

    // Scrubber Dragging
    scrubber.addEventListener("input", () => {
        const dur = getDuration();
        const targetTime = (scrubber.value / 100) * dur;
        v1.currentTime = targetTime;
        v2.currentTime = targetTime;
        timeDisplay.textContent = `${formatTime(targetTime)} / ${formatTime(dur)}`;
    });

    // Direct Time Jump
    function handleJump() {
        const val = parseFloat(timeInput.value);
        if (!isNaN(val)) {
            syncTimeTo(val);
        }
    }
    jumpBtn.addEventListener("click", handleJump);
    timeInput.addEventListener("keydown", (e) => { if (e.key === "Enter") handleJump(); });

    // Update scrubber as video plays
    v1.addEventListener("timeupdate", () => {
        if (!v1.paused) {
            const dur = getDuration();
            scrubber.value = dur ? (v1.currentTime / dur) * 100 : 0;
            timeDisplay.textContent = `${formatTime(v1.currentTime)} / ${formatTime(dur)}`;
        }
    });

    // Sync metadata load
    v1.addEventListener("loadedmetadata", () => syncTimeTo(0));
});
</script>
```