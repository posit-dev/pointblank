-- termshow.lua — Quarto shortcode for embedding terminal recordings.
--
-- Usage in .qmd files:
--
--   {{< termshow file="demos/getting-started" >}}
--
--   {{< termshow file="demos/install" autoplay="true" pause_on_chapters="true" >}}
--
-- Options:
--   file             (required) Path to .termshow file (without extension),
--                    or to a bundle (a .termshow.json file)
--   autoplay         "true"/"false" - start playing automatically
--   loop             "true"/"false" - loop playback
--   speed            number - playback speed multiplier
--   pause_on_chapters "true"/"false" - auto-pause at chapter boundaries
--   poster           number - time (s) of the frame shown before playback
--   controls         "true"/"false" - show control bar
--   theme            "auto"/"dark"/"light" - player theme override
--
-- Rendered output is looked up at termshow/<file>/ first (how Great Docs
-- builds lay it out), then termshow/<basename>/ (the default output of a
-- standalone `termshow render`).
--
-- The shortcode embeds the manifest inline so the player works without
-- network fetches (including with file:// protocol). A version 2 manifest
-- carries the screens themselves (the player draws them as SVG); for older
-- version 1 output the SVG frame files are inlined alongside it.
--
-- A bundle (from `great-docs termshow export --bundle`) is read relative to
-- the document, then to the project, and embedded the same way. It has no
-- SVG files, so there's no poster image before the player starts.
--
-- When the rendered output has a transcript (_transcript.md, or a bundle's
-- `transcript` member), it follows the player as HTML, hidden from view but
-- read by screen readers. The player replaces everything inside its own
-- element when it starts, so the transcript sits next to it rather than
-- inside.

local function escape_attr(s)
    if s == nil then return "" end
    -- Escape "&" first so the entities added below aren't double-escaped
    local escaped = s:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;"):gsub(">", "&gt;")
    return escaped
end

local function kwarg_str(kwargs, key)
    local raw = kwargs[key]
    if raw == nil then return "" end
    local s = pandoc.utils.stringify(raw)
    return s or ""
end

--- Read a file relative to the Quarto project input directory.
local function read_project_file(rel_path)
    -- quarto.project.directory is the project root during render
    local base = ""
    if quarto and quarto.project and quarto.project.directory then
        base = quarto.project.directory .. "/"
    end
    local f = io.open(base .. rel_path, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    return content
end

--- Read a file by its path.
local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    return content
end

--- Read a bundle relative to the document's directory, then the project's.
local function read_bundle(rel_path)
    local input = quarto.doc and quarto.doc.input_file
    local dir = input and input:match("^(.*)[/\\]")
    return (dir and read_file(dir .. "/" .. rel_path)) or read_project_file(rel_path)
end

--- Return a bundle's transcript Markdown, or nil.
local function bundle_transcript(bundle_json)
    if not (quarto.json and quarto.json.decode) then return nil end
    local ok, data = pcall(quarto.json.decode, bundle_json)
    if ok and type(data) == "table" and type(data.transcript) == "string" then
        return data.transcript
    end
    return nil
end

--- Return the keyframe file to use as the poster for a given time (seconds).
local function poster_frame(manifest_json, poster_time)
    local chosen = nil
    for t, f in manifest_json:gmatch('"time"%s*:%s*([%d%.]+)%s*,%s*"file"%s*:%s*"([^"]+)"') do
        if chosen == nil or tonumber(t) <= poster_time then
            chosen = f
        else
            break
        end
    end
    return chosen
end

--- Return the manifest's format version (1 when it can't be read).
local function manifest_version(manifest_json)
    return tonumber(manifest_json:match('^%s*{%s*"version"%s*:%s*(%d+)')) or 1
end

-- Numbers the transcripts on a page, for their ids
local transcript_count = 0

--- Return a transcript's Markdown as HTML.
---
--- Code blocks are written by hand, so they stay plain `<pre><code>` without
--- syntax highlighting markup.
local function transcript_html(markdown)
    local doc = pandoc.read(markdown, "markdown")
    doc = doc:walk({
        CodeBlock = function(block)
            local text = block.text:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;")
            return pandoc.RawBlock("html",
                '<pre><code class="language-console">' .. text .. "</code></pre>")
        end,
    })
    return pandoc.write(doc, "html")
end

--- Register CSS/JS/font assets as a Quarto HTML dependency (once per document).
local dependency_registered = false
local function ensure_dependency()
    if dependency_registered then return end
    dependency_registered = true

    quarto.doc.add_html_dependency({
        name = "termshow-player",
        version = "1.0.0",
        stylesheets = { "termshow.css" },
        scripts = { "termshow.js" },
    })
end

return {
    ["termshow"] = function(args, kwargs, meta)
        -- Ensure player assets are included in the page
        if quarto.doc and quarto.doc.add_html_dependency then
            ensure_dependency()
        end

        -- Get file path (required)
        local file = kwarg_str(kwargs, "file")
        if file == "" and #args > 0 then
            file = pandoc.utils.stringify(args[1])
        end
        if file == "" then
            quarto.log.warning("[termshow] 'file' attribute is required")
            return pandoc.Null()
        end

        -- A bundle, or the rendered output directory
        -- (a trailing ".termshow" extension is optional)
        local bundle = file:match("%.json$") ~= nil
        local rel = file:gsub("^%./", "")
        local manifest_json, basename, tp_dir
        if bundle then
            basename = (rel:match("([^/]+)$") or rel):gsub("%.termshow%.json$", ""):gsub("%.json$", "")
            manifest_json = read_bundle(rel)
            if manifest_json then
                -- Keep "</script>" in the text from ending the script element
                manifest_json = manifest_json:gsub("</", "<\\/")
            else
                quarto.log.warning("[termshow] bundle not found: '" .. file .. "'")
            end
        else
            rel = rel:gsub("%.termshow$", "")
            basename = rel:match("([^/]+)$") or rel
            tp_dir = "termshow/" .. rel .. "/"
            manifest_json = read_project_file(tp_dir .. "manifest.json")
            if not manifest_json and basename ~= rel then
                local fallback_dir = "termshow/" .. basename .. "/"
                local fallback_json = read_project_file(fallback_dir .. "manifest.json")
                if fallback_json then
                    tp_dir = fallback_dir
                    manifest_json = fallback_json
                end
            end
            if not manifest_json then
                quarto.log.warning("[termshow] no rendered output found for '" .. file
                    .. "' (looked in termshow/" .. rel .. "/). Run `great-docs termshow render`"
                    .. " or check the file path.")
            end
        end

        -- Read options
        local autoplay = kwarg_str(kwargs, "autoplay")
        if autoplay == "" then autoplay = "false" end
        local loop = kwarg_str(kwargs, "loop")
        if loop == "" then loop = "false" end
        local speed = kwarg_str(kwargs, "speed")
        if speed == "" then speed = "1" end
        local pause_on_chapters = kwarg_str(kwargs, "pause_on_chapters")
        if pause_on_chapters == "" then pause_on_chapters = "false" end
        local poster = kwarg_str(kwargs, "poster")
        if poster == "" then poster = "0" end
        local controls = kwarg_str(kwargs, "controls")
        if controls == "" then controls = "true" end
        local theme = kwarg_str(kwargs, "theme")
        if theme == "" then theme = "auto" end

        -- Version 1 manifests only reference SVG files; read them in so they
        -- can be inlined
        local frames_json = nil
        if manifest_json and not bundle and manifest_version(manifest_json) < 2 then
            frames_json = "[]"
            -- Parse keyframes from manifest to know which SVGs to embed
            local frames = {}
            for frame_file in manifest_json:gmatch('"file"%s*:%s*"([^"]+)"') do
                local svg = read_project_file(tp_dir .. frame_file)
                if svg then
                    -- Escape for JSON string embedding
                    local escaped = svg:gsub("\\", "\\\\"):gsub('"', '\\"')
                        :gsub("\n", "\\n"):gsub("\r", "\\r")
                        :gsub("\t", "\\t")
                    table.insert(frames, '"' .. escaped .. '"')
                end
            end
            if #frames > 0 then
                frames_json = "[" .. table.concat(frames, ",") .. "]"
            end
        end

        -- Build poster path (fallback if inline fails)
        local offset = ""
        if quarto and quarto.project and quarto.project.offset then
            offset = quarto.project.offset .. "/"
        end
        local poster_file = "frame-000.svg"
        if manifest_json then
            poster_file = poster_frame(manifest_json, tonumber(poster) or 0) or poster_file
        end
        local poster_path = (not bundle) and offset .. tp_dir .. poster_file
        local manifest_url = (not bundle) and offset .. tp_dir .. "manifest.json"

        -- Generate HTML with inline data
        local parts = {}
        table.insert(parts, '<div class="gd-termshow" ')
        if manifest_url then
            table.insert(parts, 'data-manifest="' .. escape_attr(manifest_url) .. '" ')
        end
        table.insert(parts, 'data-autoplay="' .. escape_attr(autoplay) .. '" ')
        table.insert(parts, 'data-loop="' .. escape_attr(loop) .. '" ')
        table.insert(parts, 'data-speed="' .. escape_attr(speed) .. '" ')
        table.insert(parts, 'data-pause-on-chapters="' .. escape_attr(pause_on_chapters) .. '" ')
        table.insert(parts, 'data-poster="' .. escape_attr(poster) .. '" ')
        table.insert(parts, 'data-controls="' .. escape_attr(controls) .. '" ')
        table.insert(parts, 'data-theme="' .. escape_attr(theme) .. '">')

        -- Embed manifest as inline JSON script
        if manifest_json then
            table.insert(parts, '\n  <script type="application/json" class="gd-tp-manifest">')
            table.insert(parts, manifest_json)
            table.insert(parts, '</script>')
            if frames_json then
                -- Embed SVG frames as inline JSON array
                table.insert(parts, '\n  <script type="application/json" class="gd-tp-frames">')
                table.insert(parts, frames_json)
                table.insert(parts, '</script>')
            end
        end

        -- Poster image (visible before JS loads, works with file://)
        if poster_path then
            table.insert(parts, '\n  <img src="' .. escape_attr(poster_path) .. '" ')
            table.insert(parts, 'alt="Terminal recording: ' .. escape_attr(basename) .. '" ')
            table.insert(parts, 'class="gd-termshow-poster" loading="lazy"/>')
        end

        table.insert(parts, '\n  <noscript>')
        table.insert(parts, '\n    <p>Terminal recording: ' .. escape_attr(basename))
        table.insert(parts, ' (requires JavaScript for playback)</p>')
        table.insert(parts, '\n  </noscript>')
        table.insert(parts, '\n</div>')

        local transcript = nil
        if manifest_json then
            transcript = bundle and bundle_transcript(manifest_json)
                or (not bundle and read_project_file(tp_dir .. "_transcript.md"))
        end
        if transcript and transcript:match("%S") then
            transcript_count = transcript_count + 1
            local id = "gd-termshow-transcript-" .. transcript_count
            table.insert(parts, '\n<div class="gd-termshow-transcript" role="region" ')
            table.insert(parts, 'aria-labelledby="' .. id .. '">')
            table.insert(parts, '\n<p id="' .. id .. '">Transcript of the terminal recording: ')
            table.insert(parts, escape_attr(basename) .. '</p>\n')
            table.insert(parts, transcript_html(transcript))
            table.insert(parts, '\n</div>')
        end

        return pandoc.RawInline("html", table.concat(parts))
    end
}
