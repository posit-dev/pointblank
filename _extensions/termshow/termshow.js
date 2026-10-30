/**
 * Great Docs Termshow Player
 *
 * A lightweight player for pre-rendered SVG terminal recordings.
 * Loads a manifest.json and displays keyframe SVGs with timeline navigation,
 * chapter markers, annotations, and keyboard controls.
 *
 * A version 2 manifest stores each unique screen as rows of styled spans and
 * the player draws the SVG itself (`renderFrameSvg`); version 1 manifests
 * point at pre-rendered SVG files, which are inlined or fetched.
 *
 * A manifest rendered with a light/dark theme pair carries a `dark` member
 * with the dark theme's colors and rows; the player draws whichever side
 * matches the page's color mode and redraws when the mode changes.
 *
 * Players come from two places: `<div class="gd-termshow" data-manifest=...>`
 * containers written by the Quarto shortcode, and `<termshow-player src=...>`
 * elements, which work on any HTML page.
 */
(function () {
  'use strict';

  const PLAYER_CLASS = 'gd-termshow';
  const ACTIVE_CLASS = 'gd-tp-active';
  const ELEMENT_NAME = 'termshow-player';

  /**
   * Fetch a resource as text or JSON, with file:// fallback via XHR.
   */
  function fetchResource(url, asJson) {
    if (window.location.protocol === 'file:') {
      return new Promise(function (resolve, reject) {
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.onload = function () {
          if (xhr.status === 0 || xhr.status === 200) {
            resolve(asJson ? JSON.parse(xhr.responseText) : xhr.responseText);
          } else {
            reject(new Error('XHR ' + xhr.status));
          }
        };
        xhr.onerror = function () { reject(new Error('XHR network error')); };
        xhr.send();
      });
    }
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return asJson ? r.json() : r.text();
    });
  }

  /**
   * Initialize all shortcode player containers on the page.
   */
  function initAll() {
    const players = document.querySelectorAll(`.${PLAYER_CLASS}`);
    players.forEach((el) => initContainer(el));
  }

  /**
   * Initialize a shortcode container, which carries its options as data-*
   * attributes. `<termshow-player>` elements set themselves up.
   */
  function initContainer(container) {
    if (container.localName === ELEMENT_NAME) return;
    // A bundle embedded by the shortcode has only its inline manifest
    const manifestUrl = container.dataset.manifest || null;
    const inline = readInline(container);
    if (!manifestUrl && !inline.manifest) return;
    initPlayer(
      container,
      manifestUrl,
      readOptions((name) => container.getAttribute('data-' + name)),
      inline
    );
  }

  /**
   * Read player options through `get(name)`, which returns an attribute's
   * value or null. Flags are on when present, unless set to "false".
   */
  function readOptions(get) {
    const flag = (name, fallback) => {
      const value = get(name);
      return value == null ? fallback : value !== 'false';
    };
    return {
      autoplay: flag('autoplay', false),
      speed: parseFloat(get('speed')) || 1.0,
      pauseOnChapters: flag('pause-on-chapters', false),
      loop: flag('loop', false),
      poster: parseFloat(get('poster')) || 0,
      controls: flag('controls', true),
      theme: get('theme') || 'auto',
    };
  }

  /**
   * Read the manifest (and version 1 frames) embedded in a container as
   * `<script type="application/json">` elements.
   */
  function readInline(container) {
    const parse = (selector) => {
      const el = container.querySelector(selector);
      if (!el) return null;
      try { return JSON.parse(el.textContent); } catch (e) { return null; }
    };
    return { manifest: parse('script.gd-tp-manifest'), frames: parse('script.gd-tp-frames') };
  }

  /**
   * The manifest URL for a `src` attribute: a `.json` file as given, or a
   * render directory's `manifest.json`.
   */
  function manifestUrlFor(src) {
    if (!src) return null;
    if (/\.json([?#]|$)/.test(src)) return src;
    return src.replace(/\/?$/, '/') + 'manifest.json';
  }

  // Readers who ask for reduced motion: players don't autoplay or loop, they
  // stop at each chapter, and overlays appear without fading
  const motionQuery = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;

  function reducedMotion() {
    return !!(motionQuery && motionQuery.matches);
  }

  /**
   * Opacity of an overlay `elapsed` seconds into a showing of `length`
   * seconds, fading in and out over the given times (none with reduced motion).
   */
  function fadeOpacity(elapsed, length, fadeIn, fadeOut) {
    if (reducedMotion()) return 1;
    const inPart = fadeIn > 0 ? Math.min(1, elapsed / fadeIn) : 1;
    const outPart = fadeOut > 0 ? Math.min(1, (length - elapsed) / fadeOut) : 1;
    return Math.min(inPart, outPart);
  }

  /** Fire a media-style event (`ready`, `play`, `pause`, `ended`) on a player. */
  function emit(container, type) {
    if (container && container.dispatchEvent && typeof Event === 'function') {
      container.dispatchEvent(new Event(type));
    }
  }

  /**
   * Initialize a single termshow player instance.
   *
   * The manifest is fetched from `manifestUrl` when there is one, falling back
   * to `inline` data; with no URL, the inline manifest is used directly.
   * Returns a controller for the player, or null if it's already running.
   */
  function initPlayer(container, manifestUrl, options, inline) {
    // Prevent double-init
    if (container.classList.contains(ACTIVE_CLASS)) return null;
    container.classList.add(ACTIVE_CLASS);

    // Player chrome theme: "dark"/"light" force it; "auto" is resolved by
    // `applyMode()` once the manifest is loaded
    if (options.theme === 'dark' || options.theme === 'light') {
      container.classList.add('gd-tp-' + options.theme);
    }

    // Create player state
    const state = {
      manifest: null,
      frames: null,
      currentTime: 0,
      playing: false,
      ended: false,
      chapterPaused: false,
      currentKeyframeIdx: -1,
      animFrameId: null,
      lastTick: null,
      speed: options.speed,
      // 'light' or 'dark': the chrome, and the side drawn for a theme pair
      mode: null,
      shownIdx: -1,
      baseUrl: manifestUrl ? manifestUrl.substring(0, manifestUrl.lastIndexOf('/') + 1) : '',
      destroyed: false,
    };

    // Title bar (separate from viewport, above it)
    const chapterBar = document.createElement('div');
    chapterBar.className = 'gd-tp-chapter-bar';

    // With reduced motion the player waits for the reader, showing the poster
    const autoplay = options.autoplay && !reducedMotion();

    // Build player DOM
    const viewport = document.createElement('div');
    viewport.className = 'gd-tp-viewport';
    // Defer tabindex until user interaction to prevent Safari from treating
    // auto-playing players as scroll-restoration targets. Without a control
    // bar the viewport is the only way to pause, so it's always focusable.
    if (!autoplay || !options.controls) {
      viewport.setAttribute('tabindex', '0');
    }
    viewport.setAttribute('role', 'group');
    viewport.setAttribute('aria-label', 'Terminal recording');

    const svgContainer = document.createElement('div');
    svgContainer.className = 'gd-tp-svg';
    viewport.appendChild(svgContainer);

    const highlightLayer = document.createElement('div');
    highlightLayer.className = 'gd-tp-highlights';
    viewport.appendChild(highlightLayer);
    state.highlightLayer = highlightLayer;

    const annotationLayer = document.createElement('div');
    annotationLayer.className = 'gd-tp-annotations';
    viewport.appendChild(annotationLayer);

    // Copy-snippet buttons (placed in chapter/title bar)
    const snippetLayer = document.createElement('div');
    snippetLayer.className = 'gd-tp-snippets';
    chapterBar.appendChild(snippetLayer);
    state.snippetLayer = snippetLayer;

    // Center overlay (play / replay)
    const centerOverlay = document.createElement('div');
    centerOverlay.className = 'gd-tp-center-overlay';
    centerOverlay.setAttribute('aria-hidden', 'true');
    const centerBtn = document.createElement('div');
    centerBtn.className = 'gd-tp-center-overlay-btn';
    centerBtn.innerHTML = '<span class="gd-tp-icon-play">\u25b6</span>';
    centerOverlay.appendChild(centerBtn);
    viewport.appendChild(centerOverlay);

    const controls = buildControls();

    // Read out by screen readers: chapter stops and the end of the recording
    const status = document.createElement('div');
    status.className = 'gd-tp-status';
    status.setAttribute('role', 'status');
    status.setAttribute('aria-live', 'polite');

    // Replace container content with player structure
    container.innerHTML = '';
    container.appendChild(chapterBar);
    container.appendChild(viewport);
    container.appendChild(controls.root);
    container.appendChild(status);
    // For the playback functions outside this one
    state.announce = announce;
    state.focusViewport = focusViewport;
    if (!options.controls) {
      controls.root.style.display = 'none';
      container.classList.add('gd-tp-no-controls');
    }

    // Initialize with manifest (inline or fetched)
    function activate(manifest, frames) {
      if (state.destroyed) return;
      state.manifest = manifest;
      state.frames = frames;
      state.svgCache = {};
      applyMode();
      if (options.theme !== 'dark' && options.theme !== 'light') {
        modeListeners.push(applyMode);
        watchColorMode();
      }
      controls.duration.textContent = formatTime(manifest.duration);
      controls.timeline.setAttribute('aria-valuemax', String(Math.round(manifest.duration)));
      if (manifest.title) viewport.setAttribute('aria-label', 'Terminal recording: ' + manifest.title);
      renderChapterMarkers(controls.timeline, manifest);

      // Add traffic light decorations if window_chrome is set
      var chrome = manifest.window_chrome || 'none';
      var hasChapters = manifest.chapters && manifest.chapters.length > 0;
      var hasSnippets = manifest.snippets && manifest.snippets.length > 0;
      if (chrome === 'none' && !hasChapters && !hasSnippets) {
        chapterBar.style.display = 'none';
      } else if (chrome !== 'none') {
        chapterBar.classList.add('gd-tp-has-chrome');
        var lights = document.createElement('div');
        lights.className = 'gd-tp-traffic-lights' + (chrome === 'minimal' ? ' gd-tp-chrome-minimal' : '');
        lights.setAttribute('aria-hidden', 'true');
        lights.innerHTML = '<span class="gd-tp-traffic-dot gd-tp-dot-close"></span>' +
          '<span class="gd-tp-traffic-dot gd-tp-dot-minimize"></span>' +
          '<span class="gd-tp-traffic-dot gd-tp-dot-maximize"></span>';
        chapterBar.appendChild(lights);
      }

      // Add a span for chapter text (so traffic lights aren't overwritten)
      var chapterText = document.createElement('span');
      chapterText.className = 'gd-tp-chapter-text';
      chapterBar.appendChild(chapterText);
      state.chapterText = chapterText;

      // The element shows a bundle's transcript (the shortcode writes its own)
      if (options.transcript && typeof manifest.transcript === 'string' && manifest.transcript.trim()) {
        container.appendChild(buildTranscript(manifest));
      }

      updateChapterBar(chapterBar, manifest, 0);
      // Show the poster frame (playback still starts from 0)
      showFrame(state, svgContainer, findKeyframeIndex(manifest.keyframes, options.poster));
      updateSlider(state, controls);
      emit(container, 'ready');
      if (autoplay) {
        play(state, svgContainer, annotationLayer, controls, options, container, centerOverlay, chapterBar);
      }
    }

    // Set the chrome classes for the current color mode, and redraw a theme
    // pair's frame when the mode changes
    function applyMode() {
      var mode = colorMode(state.manifest, options.theme);
      if (mode === state.mode) return;
      var redraw = state.mode !== null && state.manifest && state.manifest.dark;
      state.mode = mode;
      container.classList.toggle('gd-tp-dark', mode === 'dark');
      container.classList.toggle('gd-tp-light', mode === 'light');
      if (redraw && state.shownIdx >= 0) {
        showFrame(state, svgContainer, state.shownIdx);
      }
    }

    if (manifestUrl) {
      // Always try fetching the external manifest first (it reflects the latest
      // termshow render without requiring an HTML rebuild). Fall back to inline
      // data if the fetch fails (file:// protocol, offline, etc.).
      fetchResource(manifestUrl, true).then(
        function (manifest) { activate(manifest, null); },
        function (err) {
          if (inline.manifest) activate(inline.manifest, inline.frames);
          else console.warn('[termshow] Failed to load manifest:', manifestUrl, err);
        }
      );
    } else if (inline.manifest) {
      activate(inline.manifest, inline.frames);
    }

    // Leave the ended state, so the player can play or seek again
    function clearEnded() {
      state.ended = false;
      controls.playBtn.textContent = '\u25b6';
      controls.playBtn.setAttribute('aria-label', 'Play');
      centerBtn.innerHTML = '<span class="gd-tp-icon-play">\u25b6</span>';
      container.classList.remove('gd-tp-ended');
    }

    // Leave a chapter stop, so the center overlay can show again
    function clearChapterPause() {
      state.chapterPaused = false;
      container.classList.remove('gd-tp-chapter-paused');
    }

    // After the end, go back to the first frame without playing
    function rewind() {
      clearEnded();
      state.currentTime = 0;
      state.currentKeyframeIdx = -1;
      updateDisplay(state, svgContainer, annotationLayer, controls);
      updateChapterBar(chapterBar, state.manifest, state.currentTime);
    }

    function startPlaying() {
      play(state, svgContainer, annotationLayer, controls, options, container, centerOverlay, chapterBar);
    }

    // Play, pause, or (at the end) rewind: what the play button does
    function togglePlay() {
      if (!state.manifest) return;
      ensureTabindex();
      if (state.playing) pause(state, controls, container, centerOverlay);
      else if (state.ended) rewind();
      else startPlaying();
    }

    function seekTo(time) {
      if (state.ended) clearEnded();
      clearChapterPause();
      seek(state, time, svgContainer, annotationLayer, controls);
      updateChapterBar(chapterBar, state.manifest, state.currentTime);
    }

    // Jump to a chapter's start (if there is one), and say which chapter it is
    function seekToChapter(time) {
      if (time == null) return;
      seekTo(time);
      var label = chapterAt(state.manifest, state.currentTime);
      if (label) announce('Chapter: ' + label);
    }

    // Read a message out once (a repeat is changed slightly so it's read again)
    function announce(text) {
      status.textContent = text === status.textContent ? text + ' ' : text;
    }

    // Restore tabindex on first user interaction (for keyboard nav)
    function ensureTabindex() {
      if (!viewport.hasAttribute('tabindex')) {
        viewport.setAttribute('tabindex', '0');
      }
    }

    // Move focus to the viewport (when the focused control goes away)
    function focusViewport() {
      ensureTabindex();
      viewport.focus();
    }

    // Event listeners
    controls.playBtn.addEventListener('click', togglePlay);

    centerOverlay.addEventListener('click', () => {
      ensureTabindex();
      if (state.ended) rewind();
      else startPlaying();
    });

    // Double-click/tap on viewport: left 30% rewinds 5s, right 30% fast-forwards 5s
    // Uses pointerdown for immediate, reliable detection unaffected by overlay state
    var vpTapTime = 0;
    var vpTapX = 0;

    function handleViewportSeek(clientX) {
      if (!state.manifest) return;
      var rect = viewport.getBoundingClientRect();
      var x = clientX - rect.left;
      var w = rect.width;
      var newTime;
      if (x < w * 0.3) {
        newTime = Math.max(0, state.currentTime - 5);
      } else if (x > w * 0.7) {
        newTime = Math.min(state.manifest.duration, state.currentTime + 5);
      } else {
        return; // Middle 40% — no action
      }
      if (state.ended) clearEnded();
      seek(state, newTime, svgContainer, annotationLayer, controls);
      updateChapterBar(chapterBar, state.manifest, state.currentTime);
    }

    viewport.addEventListener('pointerdown', (e) => {
      if (!state.manifest) return;
      if (e.target.closest('.gd-tp-controls')) return;
      if (!state.playing && !state.ended) return; // Let center overlay handle paused state
      var now = Date.now();
      var dx = Math.abs(e.clientX - vpTapX);
      if (now - vpTapTime < 400 && dx < 50) {
        vpTapTime = 0;
        handleViewportSeek(e.clientX);
      } else {
        vpTapTime = now;
        vpTapX = e.clientX;
      }
    });

    controls.timeline.addEventListener('mousedown', (e) => {
      if (!state.manifest) return;
      // If the click landed on a chapter marker, snap to it instead of starting drag
      var markerEl = e.target.closest('.gd-tp-chapter-marker');
      if (markerEl) {
        var chapterTime = parseFloat(markerEl.dataset.time);
        if (!isNaN(chapterTime)) seekTo(chapterTime);
        return;
      }
      e.preventDefault();
      var wasPlaying = state.playing;
      if (state.playing) {
        pause(state, controls, container, centerOverlay);
      }
      controls.timeline.classList.add('gd-tp-dragging');
      // Reset ended/chapter-paused state
      if (state.ended) clearEnded();
      clearChapterPause();

      function seekFromEvent(ev) {
        var rect = controls.timeline.getBoundingClientRect();
        var ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / rect.width));
        seek(state, ratio * state.manifest.duration, svgContainer, annotationLayer, controls);
        updateChapterBar(chapterBar, state.manifest, state.currentTime);
      }

      seekFromEvent(e);

      function onMove(ev) {
        ev.preventDefault();
        seekFromEvent(ev);
      }

      function onUp() {
        controls.timeline.classList.remove('gd-tp-dragging');
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        if (wasPlaying) startPlaying();
      }

      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });

    // Touch support for timeline drag
    controls.timeline.addEventListener('touchstart', (e) => {
      if (!state.manifest) return;
      e.preventDefault();
      var wasPlaying = state.playing;
      if (state.playing) {
        pause(state, controls, container, centerOverlay);
      }
      controls.timeline.classList.add('gd-tp-dragging');
      if (state.ended) clearEnded();
      clearChapterPause();

      function seekFromTouch(ev) {
        var touch = ev.touches[0] || ev.changedTouches[0];
        var rect = controls.timeline.getBoundingClientRect();
        var ratio = Math.max(0, Math.min(1, (touch.clientX - rect.left) / rect.width));
        seek(state, ratio * state.manifest.duration, svgContainer, annotationLayer, controls);
        updateChapterBar(chapterBar, state.manifest, state.currentTime);
      }

      seekFromTouch(e);

      function onTouchMove(ev) {
        ev.preventDefault();
        seekFromTouch(ev);
      }

      function onTouchEnd() {
        controls.timeline.classList.remove('gd-tp-dragging');
        document.removeEventListener('touchmove', onTouchMove);
        document.removeEventListener('touchend', onTouchEnd);
        if (wasPlaying) startPlaying();
      }

      document.addEventListener('touchmove', onTouchMove, { passive: false });
      document.addEventListener('touchend', onTouchEnd);
    }, { passive: false });

    // Speed control
    controls.speedBtn.addEventListener('click', () => {
      const speeds = [0.5, 1, 1.5, 2, 3];
      const idx = speeds.indexOf(state.speed);
      state.speed = speeds[(idx + 1) % speeds.length];
      setSpeedLabel(controls.speedBtn, state.speed);
    });
    setSpeedLabel(controls.speedBtn, state.speed);

    // Keyboard shortcuts, for keys pressed while focus is anywhere in the
    // player (keys pressed elsewhere on the page are left alone)
    container.addEventListener('keydown', (e) => {
      if (!state.manifest || e.altKey || e.ctrlKey || e.metaKey) return;
      const target = e.target;
      const onSlider = target === controls.timeline;
      const onButton = !!(target.closest && target.closest('button'));
      const chapters = state.manifest.chapters || [];

      switch (e.key) {
        case ' ':
          // A focused button handles Space itself
          if (onButton) return;
          togglePlay();
          break;
        case 'ArrowRight':
          seekTo(state.currentTime + 5);
          break;
        case 'ArrowLeft':
          seekTo(state.currentTime - 5);
          break;
        case 'ArrowUp':
          // Arrows up and down scroll the page, except on the slider
          if (!onSlider) return;
          seekTo(state.currentTime + 5);
          break;
        case 'ArrowDown':
          if (!onSlider) return;
          seekTo(state.currentTime - 5);
          break;
        case 'Home':
          seekTo(0);
          break;
        case 'End':
          seekTo(state.manifest.duration);
          break;
        case 'PageUp':
          if (!onSlider) return;
          seekToChapter(nextChapterTime(chapters, state.currentTime));
          break;
        case 'PageDown':
          if (!onSlider) return;
          seekToChapter(prevChapterTime(chapters, state.currentTime));
          break;
        case ']':
        case '.':
          seekToChapter(nextChapterTime(chapters, state.currentTime));
          break;
        case '[':
        case ',':
          seekToChapter(prevChapterTime(chapters, state.currentTime));
          break;
        default:
          return;
      }
      e.preventDefault();
    });

    // Controller for scripting the player (the `<termshow-player>` methods)
    return {
      play() {
        if (!state.manifest || state.playing) return;
        if (state.ended) {
          clearEnded();
          state.currentTime = 0;
          state.currentKeyframeIdx = -1;
        }
        startPlaying();
      },
      pause() {
        if (state.playing) pause(state, controls, container, centerOverlay);
      },
      seek(time) {
        if (!state.manifest) return;
        seekTo(Number(time) || 0);
      },
      get currentTime() { return state.currentTime; },
      get duration() { return state.manifest ? state.manifest.duration : NaN; },
      get paused() { return !state.playing; },
      // Stop playback and remove the player, leaving an empty container
      destroy() {
        state.destroyed = true;
        state.playing = false;
        if (state.animFrameId) cancelAnimationFrame(state.animFrameId);
        const at = modeListeners.indexOf(applyMode);
        if (at >= 0) modeListeners.splice(at, 1);
        container.innerHTML = '';
        Array.from(container.classList)
          .filter((name) => name.indexOf('gd-tp-') === 0)
          .forEach((name) => container.classList.remove(name));
      },
    };
  }

  // --- Playback ---

  function play(state, svgContainer, annotationLayer, controls, options, container, centerOverlay, chapterBar) {
    const wasPlaying = state.playing;
    state.playing = true;
    state.ended = false;
    state.chapterPaused = false;
    state.lastTick = performance.now();
    controls.playBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none"><rect x="6" y="4" width="4" height="16" rx="1"/><rect x="14" y="4" width="4" height="16" rx="1"/></svg>';
    controls.playBtn.setAttribute('aria-label', 'Pause');
    container.classList.add('gd-tp-playing');
    container.classList.remove('gd-tp-ended');
    container.classList.remove('gd-tp-chapter-paused');

    function tick(now) {
      if (!state.playing) return;

      const dt = ((now - state.lastTick) / 1000) * state.speed;
      state.lastTick = now;
      state.currentTime += dt;

      const manifest = state.manifest;
      if (state.currentTime >= manifest.duration) {
        if (options.loop && !reducedMotion()) {
          state.currentTime = 0;
        } else {
          state.currentTime = manifest.duration;
          state.ended = true;
          pause(state, controls, container, centerOverlay);
          // Show replay icon in controls and overlay
          controls.playBtn.textContent = '\u21ba'; // ↺
          controls.playBtn.setAttribute('aria-label', 'Replay');
          centerOverlay.querySelector('.gd-tp-center-overlay-btn').innerHTML =
            '<span>\u21ba</span>';
          container.classList.remove('gd-tp-playing');
          container.classList.add('gd-tp-ended');
          updateSlider(state, controls);
          if (state.announce) state.announce('Recording ended');
          emit(container, 'ended');
          return;
        }
      }

      // Check chapter pause (with reduced motion, playback steps from chapter
      // to chapter)
      if ((options.pauseOnChapters || reducedMotion()) && manifest.chapters.length > 0) {
        const prevTime = state.currentTime - dt;
        for (const ch of manifest.chapters) {
          if (ch.time > 0 && prevTime < ch.time && state.currentTime >= ch.time) {
            state.currentTime = ch.time;
            state.chapterPaused = true;
            pause(state, controls, container, centerOverlay);
            updateDisplay(state, svgContainer, annotationLayer, controls);
            updateChapterBar(chapterBar, manifest, state.currentTime);
            if (state.announce) {
              state.announce('Paused at chapter: ' + (ch.label || 'untitled') + '. Press Play to go on.');
            }
            return;
          }
        }
      }

      updateDisplay(state, svgContainer, annotationLayer, controls);
      updateChapterBar(chapterBar, manifest, state.currentTime);
      state.animFrameId = requestAnimationFrame(tick);
    }

    state.animFrameId = requestAnimationFrame(tick);
    if (!wasPlaying) emit(container, 'play');
  }

  function pause(state, controls, container, centerOverlay) {
    const wasPlaying = state.playing;
    state.playing = false;
    if (state.animFrameId) {
      cancelAnimationFrame(state.animFrameId);
      state.animFrameId = null;
    }
    if (!state.ended) {
      controls.playBtn.textContent = '\u25b6'; // ▶
      controls.playBtn.setAttribute('aria-label', 'Play');
    }
    if (container) {
      container.classList.remove('gd-tp-playing');
      // Hide center overlay on chapter pauses (only show at start/end)
      if (state.chapterPaused) {
        container.classList.add('gd-tp-chapter-paused');
      } else {
        container.classList.remove('gd-tp-chapter-paused');
      }
    }
    if (wasPlaying) emit(container, 'pause');
  }

  function seek(state, time, svgContainer, annotationLayer, controls) {
    state.currentTime = Math.max(0, Math.min(time, state.manifest.duration));
    updateDisplay(state, svgContainer, annotationLayer, controls);
  }

  /** Start of the next chapter after `time`, or null after the last one. */
  function nextChapterTime(chapters, time) {
    for (const ch of chapters) {
      if (ch.time > time + 0.1) return ch.time;
    }
    return null;
  }

  /**
   * Start of the chapter before `time` (a little way into a chapter, its own
   * start), or 0 before the first one. Null when there are no chapters.
   */
  function prevChapterTime(chapters, time) {
    if (!chapters.length) return null;
    for (let i = chapters.length - 1; i >= 0; i--) {
      if (chapters[i].time < time - 0.5) return chapters[i].time;
    }
    return 0;
  }

  // --- Display ---

  function updateDisplay(state, svgContainer, annotationLayer, controls) {
    const manifest = state.manifest;
    if (!manifest) return;

    // Update timeline progress
    const progress = manifest.duration > 0 ? state.currentTime / manifest.duration : 0;
    controls.progress.style.width = (progress * 100) + '%';
    controls.time.textContent = formatTime(state.currentTime);
    controls.duration.textContent = formatTime(manifest.duration - state.currentTime);
    updateSlider(state, controls);

    // Find the correct keyframe for current time
    const kfIdx = findKeyframeIndex(manifest.keyframes, state.currentTime);
    if (kfIdx !== state.currentKeyframeIdx) {
      state.currentKeyframeIdx = kfIdx;
      showFrame(state, svgContainer, kfIdx);
    }

    // Update annotations
    updateAnnotations(annotationLayer, manifest.annotations, state.currentTime);

    // Update highlights
    if (state.highlightLayer) {
      updateHighlights(state, state.currentTime, svgContainer);
    }

    // Update copy-snippet buttons
    updateSnippets(state, state.snippetLayer, manifest.snippets, state.currentTime, svgContainer);
  }

  /**
   * Keep the timeline's slider values current. They change once a second at
   * most, so screen readers aren't flooded during playback.
   */
  function updateSlider(state, controls) {
    const manifest = state.manifest;
    if (!manifest) return;
    const second = Math.floor(state.currentTime);
    const label = chapterAt(manifest, state.currentTime);
    if (second === state.sliderSecond && label === state.sliderChapter) return;
    state.sliderSecond = second;
    state.sliderChapter = label;
    const timeline = controls.timeline;
    timeline.setAttribute('aria-valuenow', String(second));
    timeline.setAttribute(
      'aria-valuetext',
      formatTime(state.currentTime) + ' of ' + formatTime(manifest.duration) +
        (label ? ', chapter: ' + label : '')
    );
  }

  /** The label of the chapter playing at `time` ("" before the first one). */
  function chapterAt(manifest, time) {
    const chapters = (manifest && manifest.chapters) || [];
    for (let i = chapters.length - 1; i >= 0; i--) {
      if (chapters[i].time <= time + 0.05) return chapters[i].label || '';
    }
    return '';
  }

  function updateChapterBar(chapterBar, manifest, time) {
    var textEl = chapterBar.querySelector('.gd-tp-chapter-text');
    if (!textEl) return;
    textEl.textContent = chapterAt(manifest, time);
  }

  function showFrame(state, svgContainer, idx) {
    const manifest = state.manifest;
    if (!manifest || idx < 0 || idx >= manifest.keyframes.length) return;

    // Pin height before replacing innerHTML to prevent layout collapse.
    // All frames in a recording share the same dimensions, so we pin once
    // and keep it for the lifetime of playback. This prevents Safari's
    // scroll-preservation heuristic from firing on the last player where
    // even a sub-frame height flicker affects scrollHeight.
    if (!svgContainer.style.minHeight) {
      var pinHeight = svgContainer.offsetHeight;
      if (pinHeight > 0) {
        svgContainer.style.minHeight = pinHeight + 'px';
      }
    }

    const kf = manifest.keyframes[idx];
    state.shownIdx = idx;

    // Version 2: draw the screen from the manifest
    if (manifest.frames && kf.frame != null) {
      const mode = manifest.dark && state.mode === 'dark' ? 'dark' : 'light';
      const key = mode + kf.frame;
      if (!state.svgCache[key]) {
        state.svgCache[key] = renderFrameSvg(manifest, kf.frame, mode);
      }
      svgContainer.innerHTML = state.svgCache[key];
      return;
    }

    // Use inline frames if available (no fetch needed)
    if (state.frames && state.frames[idx]) {
      svgContainer.innerHTML = state.frames[idx];
      return;
    }

    // Fallback: fetch the SVG file
    const svgUrl = state.baseUrl + kf.file;

    fetchResource(svgUrl, false)
      .then((svg) => {
        svgContainer.innerHTML = svg;
      })
      .catch(() => {});
  }

  function escapeXml(text) {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  // --- Transcripts ---

  let transcriptCount = 0;

  // A visually hidden region with a bundle's transcript, for screen readers
  function buildTranscript(manifest) {
    const region = document.createElement('div');
    region.className = 'gd-termshow-transcript';
    region.setAttribute('role', 'region');
    const id = 'gd-tp-transcript-' + (++transcriptCount);
    region.setAttribute('aria-labelledby', id);
    region.innerHTML = '<p id="' + id + '">' +
      escapeXml('Transcript of the terminal recording' + (manifest.title ? ': ' + manifest.title : '')) +
      '</p>\n' + transcriptHtml(manifest.transcript);
    return region;
  }

  /**
   * HTML for a transcript's Markdown, which uses only what `great-docs`
   * writes: bold chapter labels, notes as quotes, and the terminal's text in
   * `console` code blocks (fenced with as many backticks as it needs).
   */
  function transcriptHtml(markdown) {
    const lines = String(markdown).replace(/\r\n?/g, '\n').split('\n');
    const blockStart = /^(`{3,}|>)/;
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const fence = lines[i].match(/^`{3,}/);
      if (fence) {
        const body = [];
        i++;
        while (i < lines.length && lines[i] !== fence[0]) body.push(lines[i++]);
        i++;
        out.push('<pre><code class="language-console">' + escapeXml(body.join('\n')) + '</code></pre>');
      } else if (lines[i].startsWith('>')) {
        const body = [];
        while (i < lines.length && lines[i].startsWith('>')) body.push(lines[i++].replace(/^> ?/, ''));
        out.push('<blockquote>' + transcriptParagraphs(body.join('\n')) + '</blockquote>');
      } else if (lines[i].trim()) {
        const body = [];
        while (i < lines.length && lines[i].trim() && !blockStart.test(lines[i])) body.push(lines[i++]);
        out.push(transcriptParagraphs(body.join('\n')));
      } else {
        i++;
      }
    }
    return out.join('\n');
  }

  function transcriptParagraphs(text) {
    return text.split(/\n\s*\n/).filter((p) => p.trim()).map((p) => {
      const bold = p.match(/^\*\*([\s\S]+)\*\*$/);
      const html = escapeXml(unescapeMarkdown(bold ? bold[1] : p));
      return '<p>' + (bold ? '<strong>' + html + '</strong>' : html) + '</p>';
    }).join('');
  }

  function unescapeMarkdown(text) {
    return text.replace(/\\([\\`*_\[\]<>])/g, '$1');
  }

  /**
   * Draw frame `frameIdx` of a version 2 manifest as an SVG string.
   *
   * Mirrors `render_svg()` in great_docs/_term_player/renderer.py so the
   * result matches the SVG files written at render time (the highlight and
   * snippet code measures this markup). Keep the two in sync.
   *
   * With `mode` 'dark' and a manifest that has a `dark` member, the frame is
   * drawn in the dark theme's colors and rows.
   */
  function renderFrameSvg(manifest, frameIdx, mode) {
    const screen = manifest.screen;
    const dark = mode === 'dark' ? manifest.dark : null;
    const colors = dark || screen;
    const rows = dark ? dark.lines : manifest.lines;
    const frame = manifest.frames[frameIdx];
    const cols = frame.size ? frame.size[0] : manifest.term.cols;
    const fmt = (n) => n.toFixed(1);

    const cellW = screen.font_size * 0.6;
    const cellH = screen.font_size * screen.line_height;
    const pad = screen.padding;
    const svgW = cols * cellW + pad * 2;
    const svgH = frame.lines.length * cellH + pad * 2;

    const parts = [];
    parts.push(
      '<svg xmlns="http://www.w3.org/2000/svg" ' +
      'viewBox="0 0 ' + fmt(svgW) + ' ' + fmt(svgH) + '" ' +
      'width="' + fmt(svgW) + '" height="' + fmt(svgH) + '" ' +
      'role="img" aria-label="Terminal recording frame">'
    );
    parts.push(
      '<style>' +
      '.gd-tp-text{font-family:' + screen.font_family + ';font-size:' + screen.font_size + 'px;' +
      'white-space:pre;dominant-baseline:text-before-edge}' +
      '.gd-tp-bold{font-weight:700}' +
      '.gd-tp-italic{font-style:italic}' +
      '.gd-tp-underline{text-decoration:underline}' +
      '.gd-tp-strikethrough{text-decoration:line-through}' +
      '.gd-tp-dim{opacity:0.6}' +
      '@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}' +
      '.gd-tp-cursor{animation:blink 1s step-end infinite}' +
      '</style>'
    );
    parts.push(
      '<rect width="' + fmt(svgW) + '" height="' + fmt(svgH) + '" fill="' + colors.bg +
      '" rx="8" ry="8"/>'
    );
    parts.push('<g transform="translate(' + pad + ',' + pad + ')">');

    frame.lines.forEach((lineIdx, rowIdx) => {
      const line = rows[lineIdx];
      const y = fmt(rowIdx * cellH);
      for (const [col, len, color] of line[0]) {
        parts.push(
          '<rect x="' + fmt(col * cellW) + '" y="' + y + '" ' +
          'width="' + fmt(len * cellW) + '" height="' + fmt(cellH) + '" fill="' + color + '"/>'
        );
      }
      for (const [col, text, color, classes] of line[1]) {
        const cls = classes ? 'gd-tp-text ' + classes : 'gd-tp-text';
        parts.push(
          '<text x="' + fmt(col * cellW) + '" y="' + y + '" class="' + cls + '" fill="' + color +
          '">' + escapeXml(text) + '</text>'
        );
      }
    });

    if (frame.cursor) {
      parts.push(
        '<rect x="' + fmt(frame.cursor[1] * cellW) + '" y="' + fmt(frame.cursor[0] * cellH) + '" ' +
        'width="' + fmt(cellW) + '" height="' + fmt(cellH) + '" ' +
        'fill="' + (colors.cursor || colors.fg) + '" opacity="0.7" class="gd-tp-cursor"/>'
      );
    }

    parts.push('</g>');
    parts.push('</svg>');
    return parts.join('\n');
  }

  function findKeyframeIndex(keyframes, time) {
    // Find the last keyframe at or before the current time
    let idx = 0;
    for (let i = keyframes.length - 1; i >= 0; i--) {
      if (keyframes[i].time <= time) {
        idx = i;
        break;
      }
    }
    return idx;
  }

  function updateAnnotations(layer, annotations, time) {
    if (!annotations || annotations.length === 0) {
      for (var i = 0; i < layer.children.length; i++) {
        layer.children[i].style.opacity = '0';
      }
      return;
    }

    // Pre-create elements with full styling at init time. Once created, ONLY
    // opacity is mutated per-frame. This prevents Safari from re-evaluating
    // position styles (top/bottom/margin-block) during playback, which triggers
    // its scroll-position recalculation bug.
    if (!layer._annEls) {
      layer._annEls = [];
      for (var j = 0; j < annotations.length; j++) {
        var ann0 = annotations[j];
        var el = document.createElement('div');
        var wCls = ann0.width && ann0.width !== 'medium' ? ' gd-tp-ann-w-' + ann0.width : '';
        el.className = 'gd-tp-annotation gd-tp-ann-' + ann0.position + ' gd-tp-ann-' + ann0.style + wCls;
        el.textContent = ann0.text;
        el.style.opacity = '0';
        el.setAttribute('aria-hidden', 'true');
        layer.appendChild(el);
        layer._annEls.push(el);
      }
    }

    for (var k = 0; k < annotations.length; k++) {
      var ann = annotations[k];
      var node = layer._annEls[k];
      if (time >= ann.time && time <= ann.time + ann.duration) {
        // Fade progress (for smooth entry/exit)
        node.style.opacity = fadeOpacity(time - ann.time, ann.duration, 0.3, 0.3);
      } else {
        if (node.style.opacity !== '0') node.style.opacity = '0';
      }
    }
  }

  // Per-manifest caches for pattern targets: row text by frame, match results,
  // and the fixed position of each non-tracking highlight
  const matchCaches = new WeakMap();

  function matchCache(manifest) {
    let cache = matchCaches.get(manifest);
    if (!cache) {
      cache = { rows: [], matches: new Map(), anchors: new WeakMap() };
      matchCaches.set(manifest, cache);
    }
    return cache;
  }

  /**
   * Plain text of each row of a version 2 frame. Spans are placed at their
   * columns (one character per cell) and gaps are filled with spaces.
   */
  function frameRows(manifest, frameIdx) {
    const cache = matchCache(manifest);
    if (!cache.rows[frameIdx]) {
      cache.rows[frameIdx] = manifest.frames[frameIdx].lines.map((lineIdx) => {
        const cells = [];
        for (const [col, text] of manifest.lines[lineIdx][1]) {
          Array.from(text).forEach((ch, i) => { cells[col + i] = ch; });
        }
        return Array.from(cells, (ch) => (ch === undefined ? ' ' : ch)).join('');
      });
    }
    return cache.rows[frameIdx];
  }

  /**
   * First row (from the top) where `pattern` matches, as a cell rectangle
   * covering capture group `group`. Returns null when nothing matches or the
   * pattern isn't a valid JavaScript regex.
   */
  function findMatch(rows, pattern, group) {
    let re;
    try {
      re = new RegExp(pattern, 'd');
    } catch (e) {
      try { re = new RegExp(pattern); } catch (e2) { return null; }
    }
    const g = group || 0;
    for (let row = 0; row < rows.length; row++) {
      const text = rows[row];
      const m = re.exec(text);
      if (!m || m[g] == null || m[g] === '') continue;
      let start = m.index;
      if (g) start = m.indices ? m.indices[g][0] : m.index + m[0].indexOf(m[g]);
      return {
        row: row,
        col: Array.from(text.slice(0, start)).length,
        width: Array.from(m[g]).length,
        height: 1,
      };
    }
    return null;
  }

  function matchInFrame(manifest, frameIdx, target) {
    const cache = matchCache(manifest);
    const key = frameIdx + '|' + (target.group || 0) + '|' + target.match;
    if (!cache.matches.has(key)) {
      cache.matches.set(key, findMatch(frameRows(manifest, frameIdx), target.match, target.group));
    }
    return cache.matches.get(key);
  }

  /**
   * Where a pattern-targeted highlight sits at `time`, as {row, col, width,
   * height} in cells, or null when it shouldn't be drawn.
   *
   * With `track_scroll`, the pattern is matched against the screen shown at
   * `time`, so the highlight follows the text and hides while it's off-screen.
   * Otherwise the highlight stays where the text first appears during its
   * time range. Needs a version 2 manifest.
   */
  function locateMatch(manifest, hl, time) {
    const target = hl.target || {};
    if (!target.match || !manifest.frames) return null;
    const keyframes = manifest.keyframes;

    if (target.track_scroll) {
      const kf = keyframes[findKeyframeIndex(keyframes, time)];
      return kf && kf.frame != null ? matchInFrame(manifest, kf.frame, target) : null;
    }

    const cache = matchCache(manifest);
    if (!cache.anchors.has(hl)) {
      let anchor = null;
      for (let i = findKeyframeIndex(keyframes, hl.time); i < keyframes.length; i++) {
        const kf = keyframes[i];
        if (kf.time > hl.time + hl.duration) break;
        const pos = kf.frame != null ? matchInFrame(manifest, kf.frame, target) : null;
        if (pos) {
          anchor = { time: kf.time, pos: pos };
          break;
        }
      }
      cache.anchors.set(hl, anchor);
    }
    const anchor = cache.anchors.get(hl);
    return anchor && time >= anchor.time ? anchor.pos : null;
  }

  /**
   * Cell grid of a version 2 frame in viewBox units, computed from the
   * manifest the same way `renderFrameSvg` lays the frame out.
   */
  function frameGrid(manifest, frameIdx) {
    const screen = manifest.screen;
    const frame = manifest.frames[frameIdx];
    const cols = frame.size ? frame.size[0] : manifest.term.cols;
    const cellW = screen.font_size * 0.6;
    const cellH = screen.font_size * screen.line_height;
    return {
      vbW: cols * cellW + screen.padding * 2,
      vbH: frame.lines.length * cellH + screen.padding * 2,
      cellW: cellW,
      cellH: cellH,
      translateX: screen.padding,
      translateY: screen.padding,
    };
  }

  function placeHighlight(node, grid, r) {
    var x = grid.translateX + r.col * grid.cellW;
    var y = grid.translateY + r.row * grid.cellH;
    node.style.left = (x / grid.vbW * 100) + '%';
    node.style.top = (y / grid.vbH * 100) + '%';
    node.style.width = (r.width * grid.cellW / grid.vbW * 100) + '%';
    node.style.height = (r.height * grid.cellH / grid.vbH * 100) + '%';
  }

  function updateHighlights(state, time, svgContainer) {
    var layer = state.highlightLayer;
    var manifest = state.manifest;
    var highlights = manifest.highlights;
    if (!highlights || highlights.length === 0) {
      if (layer._hlEls) {
        for (var i = 0; i < layer._hlEls.length; i++) {
          layer._hlEls[i].style.opacity = '0';
        }
      }
      return;
    }

    // Pre-create highlight elements once
    if (!layer._hlEls) {
      layer._hlEls = [];
      for (var j = 0; j < highlights.length; j++) {
        var hl = highlights[j];
        var el = document.createElement('div');
        el.className = 'gd-tp-highlight gd-tp-hl-' + hl.style;
        el.style.cssText = 'position:absolute;pointer-events:none;opacity:0;';
        if (hl.color) {
          el.style.setProperty('--hl-color', hl.color);
        }
        if (hl.pulse) {
          el.classList.add('gd-tp-hl-pulse');
        }
        // Badge content
        if ((hl.style === 'badge-before' || hl.style === 'badge-after') && (hl.badge_text || hl.badge_icon)) {
          var badge = document.createElement('span');
          badge.className = 'gd-tp-hl-badge';
          badge.textContent = (hl.badge_icon || '') + (hl.badge_text || '');
          el.appendChild(badge);
        }
        el.setAttribute('aria-hidden', 'true');
        layer.appendChild(el);
        layer._hlEls.push(el);
      }
    }

    // Grid in viewBox coordinates. Version 2 frames carry their own size (it
    // changes after a resize); version 1 SVGs are measured once
    var grid;
    var kf = manifest.keyframes[state.currentKeyframeIdx];
    if (manifest.frames && kf && kf.frame != null) {
      grid = frameGrid(manifest, kf.frame);
    } else {
      grid = layer._grid;
      if (!grid) {
        grid = measureViewBoxGrid(svgContainer);
        if (grid) layer._grid = grid;
      }
    }

    for (var k = 0; k < highlights.length; k++) {
      var h = highlights[k];
      var node = layer._hlEls[k];
      if (time >= h.time && time <= h.time + h.duration) {
        node.style.opacity = fadeOpacity(
          time - h.time,
          h.duration,
          h.fade_in != null ? h.fade_in : 0.3,
          h.fade_out != null ? h.fade_out : 0.3
        );

        // Position using percentage of viewBox (scales with SVG responsively)
        if (h.target && h.target.match) {
          var pos = grid ? locateMatch(manifest, h, time) : null;
          if (pos) {
            placeHighlight(node, grid, pos);
          } else {
            node.style.opacity = '0';
          }
        } else if (grid && h.target && h.target.region) {
          placeHighlight(node, grid, h.target.region);
        } else if (grid && h.target && h.target.lines) {
          var lines = h.target.lines;
          var minLine = Math.min.apply(null, lines);
          var maxLine = Math.max.apply(null, lines);
          var ly = grid.translateY + minLine * grid.cellH;
          var lh = (maxLine - minLine + 1) * grid.cellH;
          node.style.left = '0%';
          node.style.top = (ly / grid.vbH * 100) + '%';
          node.style.width = '100%';
          node.style.height = (lh / grid.vbH * 100) + '%';
        }
      } else {
        if (node.style.opacity !== '0') node.style.opacity = '0';
      }
    }
  }

  /**
   * Measure character grid in viewBox coordinates (SVG units).
   * Returns {vbW, vbH, cellW, cellH, translateX, translateY} or null.
   * These values are resolution-independent and never change with browser width.
   */
  function measureViewBoxGrid(svgContainer) {
    var svg = svgContainer.querySelector('svg');
    if (!svg) return null;

    // Get viewBox dimensions
    var viewBox = svg.getAttribute('viewBox');
    if (!viewBox) return null;
    var parts = viewBox.split(/\s+|,/);
    var vbW = parseFloat(parts[2]);
    var vbH = parseFloat(parts[3]);
    if (!vbW || !vbH) return null;

    // Find the <g transform="translate(...)"> that wraps text content
    var g = svg.querySelector('g[transform]');
    var translateX = 0, translateY = 0;
    if (g) {
      var m = g.getAttribute('transform').match(/translate\(\s*([\d.]+)\s*[,\s]\s*([\d.]+)\s*\)/);
      if (m) {
        translateX = parseFloat(m[1]);
        translateY = parseFloat(m[2]);
      }
    }

    // Determine cell size from cursor rect or text elements (in SVG units)
    var cursor = svg.querySelector('rect.gd-tp-cursor');
    var cellW, cellH;
    if (cursor) {
      cellW = parseFloat(cursor.getAttribute('width')) || 8.4;
      cellH = parseFloat(cursor.getAttribute('height')) || 21;
    } else {
      var textEls = svg.querySelectorAll('.gd-tp-text');
      if (textEls.length < 2) return null;
      var ys = [], xs = [];
      for (var i = 0; i < Math.min(textEls.length, 50); i++) {
        var y = parseFloat(textEls[i].getAttribute('y'));
        var x = parseFloat(textEls[i].getAttribute('x'));
        if (!isNaN(y) && ys.indexOf(y) < 0) ys.push(y);
        if (!isNaN(x) && xs.indexOf(x) < 0) xs.push(x);
      }
      ys.sort(function(a, b) { return a - b; });
      xs.sort(function(a, b) { return a - b; });
      cellH = ys.length >= 2 ? ys[1] - ys[0] : 21;
      cellW = xs.length >= 2 ? xs[1] - xs[0] : 8.4;
    }

    return {
      vbW: vbW,
      vbH: vbH,
      cellW: cellW,
      cellH: cellH,
      translateX: translateX,
      translateY: translateY
    };
  }

  // Hide a copy button; if it had focus, focus moves to the viewport rather
  // than being lost
  function hideSnippet(state, el) {
    if (el.style.display === 'none') return;
    const focused = el === document.activeElement;
    el.style.display = 'none';
    el.style.opacity = '0';
    if (focused && state.focusViewport) state.focusViewport();
  }

  function updateSnippets(state, layer, snippets, time, svgContainer) {
    if (!snippets || snippets.length === 0) {
      if (layer._cmdEls) {
        for (var i = 0; i < layer._cmdEls.length; i++) hideSnippet(state, layer._cmdEls[i]);
      }
      return;
    }

    // Pre-create snippet copy buttons at init time
    if (!layer._cmdEls) {
      layer._cmdEls = [];
      for (var j = 0; j < snippets.length; j++) {
        var snip = snippets[j];
        var el = document.createElement('button');
        el.className = 'gd-tp-copy-snippet';
        el.setAttribute('type', 'button');
        el.setAttribute('aria-label', 'Copy snippet: ' + (snip.text || snip.match));
        el.dataset.text = snip.text;
        el.dataset.match = snip.match || '';
        el.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>' +
          '<span class="gd-tp-copy-label">' + (snip.label || 'Copy') + '</span>';
        el.style.display = 'none';
        el.addEventListener('click', (function(btn, idx) {
          return function(e) {
            e.stopPropagation();
            var copyText = resolveSnippetText(snippets[idx], svgContainer);
            if (copyText) {
              navigator.clipboard.writeText(copyText).then(function() {
                btn.classList.add('gd-tp-copied');
                setTimeout(function() { btn.classList.remove('gd-tp-copied'); }, 1500);
              });
            }
          };
        })(el, j));
        layer.appendChild(el);
        layer._cmdEls.push(el);
      }
    }

    // Show only one snippet at a time (first active wins; overlaps are clipped)
    var activeIdx = -1;
    for (var k = 0; k < snippets.length; k++) {
      var c = snippets[k];
      if (activeIdx < 0 && time >= c.time && time <= c.time + c.duration) {
        activeIdx = k;
        layer._cmdEls[k].style.display = 'inline-flex';
        layer._cmdEls[k].style.opacity = fadeOpacity(time - c.time, c.duration, 0.3, 0.3);
      } else {
        hideSnippet(state, layer._cmdEls[k]);
      }
    }
  }

  /**
   * Resolve the text to copy for a snippet.
   * If `match` is set, extract text from the SVG and run the regex.
   * If regex has a capture group, use group(1); otherwise use group(0).
   * Falls back to literal `text` field.
   */
  function resolveSnippetText(snip, svgContainer) {
    if (snip.match) {
      var buffer = extractSvgText(svgContainer);
      try {
        var re = new RegExp(snip.match);
        var m = re.exec(buffer);
        if (m) return m[1] !== undefined ? m[1] : m[0];
      } catch (e) { /* invalid regex — fall through */ }
    }
    return snip.text || '';
  }

  /**
   * Extract the plain-text content of the current SVG frame.
   * Groups <text> elements by y-coordinate to reconstruct terminal lines.
   */
  function extractSvgText(svgContainer) {
    var svg = svgContainer.querySelector('svg');
    if (!svg) return '';
    var textEls = svg.querySelectorAll('.gd-tp-text');
    if (!textEls.length) return '';
    // Group by y-coordinate (row)
    var rows = {};
    for (var i = 0; i < textEls.length; i++) {
      var el = textEls[i];
      var y = parseFloat(el.getAttribute('y')) || 0;
      var x = parseFloat(el.getAttribute('x')) || 0;
      var key = Math.round(y * 10); // quantize y to avoid float drift
      if (!rows[key]) rows[key] = [];
      rows[key].push({ x: x, text: el.textContent || '' });
    }
    // Sort rows by y, spans within rows by x
    var keys = Object.keys(rows).map(Number).sort(function(a, b) { return a - b; });
    var lines = [];
    for (var k = 0; k < keys.length; k++) {
      var spans = rows[keys[k]].sort(function(a, b) { return a.x - b.x; });
      var line = '';
      for (var s = 0; s < spans.length; s++) line += spans[s].text;
      lines.push(line);
    }
    return lines.join('\n');
  }

  // --- Controls UI ---

  function buildControls() {
    const root = document.createElement('div');
    root.className = 'gd-tp-controls';

    const playBtn = document.createElement('button');
    playBtn.type = 'button';
    playBtn.className = 'gd-tp-play-btn';
    playBtn.textContent = '\u25b6'; // ▶
    playBtn.setAttribute('aria-label', 'Play');

    // The slider's value text gives the times to screen readers
    const time = document.createElement('span');
    time.className = 'gd-tp-time';
    time.textContent = '0:00';
    time.setAttribute('aria-hidden', 'true');

    const timeline = document.createElement('div');
    timeline.className = 'gd-tp-timeline';
    timeline.setAttribute('role', 'slider');
    timeline.setAttribute('tabindex', '0');
    timeline.setAttribute('aria-label', 'Playback position');
    timeline.setAttribute('aria-valuemin', '0');
    timeline.setAttribute('aria-valuemax', '0');
    timeline.setAttribute('aria-valuenow', '0');

    const progress = document.createElement('div');
    progress.className = 'gd-tp-progress';
    timeline.appendChild(progress);

    const duration = document.createElement('span');
    duration.className = 'gd-tp-duration';
    duration.textContent = '0:00';
    duration.setAttribute('aria-hidden', 'true');

    const speedBtn = document.createElement('button');
    speedBtn.type = 'button';
    speedBtn.className = 'gd-tp-speed-btn';

    root.appendChild(playBtn);
    root.appendChild(time);
    root.appendChild(timeline);
    root.appendChild(duration);
    root.appendChild(speedBtn);

    return { root, playBtn, time, timeline, progress, duration, speedBtn };
  }

  // The speed button shows the speed, and its name includes it
  function setSpeedLabel(speedBtn, speed) {
    speedBtn.textContent = speed + '\u00d7';
    speedBtn.setAttribute('aria-label', 'Playback speed: ' + speed + '\u00d7');
  }

  function renderChapterMarkers(timeline, manifest) {
    if (!manifest.chapters || manifest.chapters.length === 0) return;

    for (const ch of manifest.chapters) {
      if (ch.time > manifest.duration) continue;
      const pct = manifest.duration > 0 ? (ch.time / manifest.duration) * 100 : 0;
      const marker = document.createElement('div');
      marker.className = 'gd-tp-chapter-marker';
      marker.style.left = pct + '%';
      marker.setAttribute('title', ch.label || 'Chapter');
      marker.dataset.time = ch.time;
      // Chapters reach screen readers through the slider's value text
      marker.setAttribute('aria-hidden', 'true');
      timeline.appendChild(marker);
    }
  }

  // --- Color mode ---

  // Players following the page's color mode; each is called when it changes
  const modeListeners = [];
  let watchingColorMode = false;

  /**
   * The page's color mode. Great Docs and Quarto mark it on <html> (the
   * `data-bs-theme` attribute, or the quarto-dark/quarto-light classes);
   * outside such a site, the reader's system setting decides.
   */
  function pageColorMode() {
    const html = document.documentElement;
    const theme = html && html.getAttribute ? html.getAttribute('data-bs-theme') : null;
    if (theme === 'dark' || theme === 'light') return theme;
    for (const el of [html, document.body]) {
      if (el && el.classList) {
        if (el.classList.contains('quarto-dark')) return 'dark';
        if (el.classList.contains('quarto-light')) return 'light';
      }
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  }

  /** Whether a `#rgb` or `#rrggbb` color is dark (relative luminance below 0.5). */
  function isDarkColor(hex) {
    let h = String(hex || '').replace('#', '');
    if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
    if (!/^[0-9a-fA-F]{6}$/.test(h)) return true;
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255 < 0.5;
  }

  /**
   * Resolve a player's mode: a forced `theme` option wins; a theme pair
   * follows the page; a single theme matches the chrome to its background.
   */
  function colorMode(manifest, themeOption) {
    if (themeOption === 'dark' || themeOption === 'light') return themeOption;
    if (manifest && manifest.dark) return pageColorMode();
    if (manifest && manifest.screen) return isDarkColor(manifest.screen.bg) ? 'dark' : 'light';
    return 'dark';
  }

  /** Start watching for color mode changes (once per page). */
  function watchColorMode() {
    if (watchingColorMode) return;
    watchingColorMode = true;
    const notify = () => modeListeners.forEach((fn) => fn());
    const attrs = { attributes: true, attributeFilter: ['class', 'data-bs-theme'] };
    const modeObserver = new MutationObserver(notify);
    if (document.documentElement) modeObserver.observe(document.documentElement, attrs);
    if (document.body) modeObserver.observe(document.body, attrs);
    if (window.matchMedia) {
      const query = window.matchMedia('(prefers-color-scheme: dark)');
      if (query.addEventListener) query.addEventListener('change', notify);
    }
  }

  // --- Utilities ---

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  // --- <termshow-player> element ---

  const ELEMENT_ATTRIBUTES = [
    'src', 'autoplay', 'loop', 'speed', 'pause-on-chapters', 'poster', 'controls', 'theme',
  ];

  /**
   * Define `<termshow-player>`. Its attributes mirror the shortcode options;
   * `src` is a render directory or a manifest URL, and a manifest can also be
   * embedded as `<script type="application/json" class="gd-tp-manifest">`.
   * The player is built inside the element (no shadow DOM), so it takes the
   * page's termshow.css like shortcode players do.
   */
  function defineElement() {
    if (typeof customElements === 'undefined' || typeof HTMLElement === 'undefined') return;
    if (customElements.get(ELEMENT_NAME)) return;

    class TermshowPlayer extends HTMLElement {
      static get observedAttributes() { return ELEMENT_ATTRIBUTES; }

      connectedCallback() {
        if (this._player) return;
        // Children (an inline manifest) may not be parsed yet
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', () => {
            if (this.isConnected) this.connectedCallback();
          }, { once: true });
          return;
        }
        if (!this._inline) this._inline = readInline(this);
        this.classList.add(PLAYER_CLASS);
        const options = readOptions((name) => this.getAttribute(name));
        options.transcript = true;
        this._player = initPlayer(this, manifestUrlFor(this.getAttribute('src')), options, this._inline);
      }

      disconnectedCallback() {
        if (this._player) this._player.destroy();
        this._player = null;
      }

      // Any attribute change reloads the player with the new settings
      attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue || !this._player) return;
        this._player.destroy();
        this._player = null;
        this.connectedCallback();
      }

      play() { if (this._player) this._player.play(); }
      pause() { if (this._player) this._player.pause(); }
      seek(time) { if (this._player) this._player.seek(time); }
      get currentTime() { return this._player ? this._player.currentTime : 0; }
      set currentTime(time) { this.seek(time); }
      get duration() { return this._player ? this._player.duration : NaN; }
      get paused() { return this._player ? this._player.paused : true; }
    }

    customElements.define(ELEMENT_NAME, TermshowPlayer);
  }

  // --- Initialization ---

  // Exposed for tests and for embedding outside the Quarto shortcode
  window.Termshow = window.Termshow || {};
  window.Termshow.renderFrame = renderFrameSvg;
  window.Termshow.locateMatch = locateMatch;
  window.Termshow.colorMode = colorMode;
  window.Termshow.readOptions = readOptions;
  window.Termshow.manifestUrlFor = manifestUrlFor;
  window.Termshow.chapterAt = chapterAt;
  window.Termshow.nextChapterTime = nextChapterTime;
  window.Termshow.prevChapterTime = prevChapterTime;
  window.Termshow.fadeOpacity = fadeOpacity;
  window.Termshow.transcriptHtml = transcriptHtml;

  defineElement();

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initAll);
  } else {
    initAll();
  }

  // Also handle Quarto's dynamic content loading
  const observer = new MutationObserver((mutations) => {
    // While the page is still being parsed, a container can be seen before
    // its children (the poster, the rest of the manifest) have arrived;
    // `initAll` sets those players up once the document is complete
    if (document.readyState === 'loading') return;
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === 1) {
          if (node.classList && node.classList.contains(PLAYER_CLASS)) {
            initContainer(node);
          }
          const nested = node.querySelectorAll && node.querySelectorAll(`.${PLAYER_CLASS}`);
          if (nested) nested.forEach((el) => initContainer(el));
        }
      }
    }
  });
  // The script may run in <head>, before there's a body to watch
  observer.observe(document.body || document.documentElement, { childList: true, subtree: true });
})();
