# EchoTrace Design System 🎨

EchoTrace uses two coordinated palettes: **Forensic Noir** (dark, tactical UI) and **Forensic Lab** (light, print- and audit-first). Both prioritize high contrast, spectral-grid discipline, and readable forensic data.

## 🎨 Color Palette

### Dark mode — Forensic Noir (dashboard & tactical UI)

Token names align with the forensic branding spec for consistency across code and docs.

| Variable | Hex | Application |
| :--- | :--- | :--- |
| `BG_PRIMARY` | `#0A0A0B` | Primary interface surface |
| `BG_SECONDARY` | `#111113` | Cards, panels, grouped components |
| `BG_HERO_TOP` | `#0D0D0E` | Hero gradient start |
| `UI_BORDER` | `#1E1E20` | Structural borders, dividers, grid lines |
| `ACCENT_TRACE` | `#E8443A` | Alerts, trace accent, primary actions |
| `ACCENT_TRACE_HOVER` | `#D03530` | Button hover, pressed states |
| `ACCENT_TRACE_MID` | `#FF6B5E` | Progress fills, secondary hot highlights |
| `DATA_BRIGHT` | `#F0EDE8` | High-contrast titles and emphasized data |
| `TEXT_BODY` | `#C8C5BF` | Body copy and secondary labels |
| `TEXT_MUTED` | `#5A5A5E` | Metadata, captions, footers |
| `TEXT_SUBTLE` | `#2E2E32` | Lowest-priority UI chrome |
| `SUCCESS_VERIFIED` | `#3DBA7A` | Bonafide / verified-safe states |
| `WARNING_STATE` | `#FF9F1C` | Low confidence, quality warnings |

### Light mode — Forensic Lab (posters, PDFs, audit reports)

Use this palette for print, evidence PDFs, and any **light** dashboard or marketing surface. Reds and greens stay on-brand but are tuned for contrast on pale backgrounds.

| Variable | Hex | Application |
| :--- | :--- | :--- |
| `BG_ENVIRONMENT` | `#D0D4D9` | Page surround / app chrome outside the document |
| `BG_DOCUMENT` | `#FFFFFF` | Primary sheet / main content surface |
| `BG_ELEVATED` | `#F4F5F7` | Cards, callouts, and inset panels on white |
| `BG_CODE` | `#ECEFF1` | Monospace “receipt” blocks, logs, inline code backgrounds |
| `UI_STRUCTURE` | `#AAB0B8` | Major grid lines, figure frames, table rules |
| `UI_BORDER` | `#E2E5EA` | Hairline borders between light components |
| `UI_BORDER_STRONG` | `#CBD2DC` | Emphasis borders, focus rings (pair with offset outline) |
| `TEXT_PRIMARY` | `#0F172A` | Headings, primary narrative text |
| `TEXT_SECONDARY` | `#2D3436` | Body text, long descriptions |
| `TEXT_MUTED` | `#64748B` | Captions, axis labels, de-emphasized metadata |
| `TEXT_ON_ACCENT` | `#FFFFFF` | Label text on filled red or dark chips |
| `ALERT_FORENSIC` | `#D63031` | Anomaly markers, verdict banners, chart spikes |
| `ALERT_FORENSIC_SOFT` | `#FDE8E8` | Light wash behind alerts (badges, table rows) |
| `ACCENT_TRACE` | `#C53030` | Primary brand actions on light (slightly deeper than dark UI red) |
| `ACCENT_TRACE_HOVER` | `#A72828` | Hover / active for accent controls |
| `SUCCESS_VERIFIED` | `#1F8A54` | Verified bonafide (darker green for WCAG on white) |
| `SUCCESS_SOFT` | `#E6F7EE` | Positive status backgrounds |
| `WARNING_STATE` | `#B45309` | Warnings on light (readable on `BG_DOCUMENT`) |
| `WARNING_SOFT` | `#FFF4E5` | Warning callout backgrounds |
| `FOCUS_RING` | `#2563EB` | Keyboard focus (accessibility); use `2px solid` + offset |
| `LINK` | `#1D4ED8` | Hyperlinks on light surfaces |
| `LINK_HOVER` | `#1E40AF` | Link hover |

**Light-mode usage notes**

- Prefer **`BG_DOCUMENT`** for the main column and **`BG_ELEVATED`** only where depth is needed; avoid stacking many gray levels.
- Forensic grid on light: 1px lines at **`UI_STRUCTURE`** with **3–5% opacity**, or use **`UI_BORDER`** at full opacity with 32–40px spacing (same principle as dark Forensic Grid).
- **Verification stamps** (“FORENSIC COPY”, “AUDIT VERIFIED”): use **`ALERT_FORENSIC`** or **`TEXT_MUTED`** at ~15° rotation; keep opacity low (8–15%) so they do not fight content.

Implement light and dark as **separate token maps** (e.g. `:root` for Noir, `[data-theme="light"]` or a `.theme-forensic-lab` scope for Lab) so the same logical names—`ACCENT_TRACE`, `UI_BORDER`, etc.—resolve to the correct hex per surface.

---

## 🔡 Typography

| Font Family | Usage | Characteristics |
| :--- | :--- | :--- |
| **Bebas Neue** | Main Wordmark, Big Metrics | Bold, condensed, authoritative. `letter-spacing: 0.12em` |
| **Space Mono** | Data, Labels, Code, Reports | Monospaced, technical, forensic feel. `letter-spacing: 0.15em` to `0.22em` |
| **DM Sans** | Body Copy, Paragraphs | Clean, geometric, readable. Weights: `300`, `400`, `500`, `600`. |

---

## 📐 Structural Elements

### Border Radius
EchoTrace avoids overly rounded corners to maintain a rigid, technical feel.
- **Large Containers & Cards:** `4px`
- **Buttons, Alerts & Stat Pills:** `2px`
- **Progress Bars & Inner Fills:** `1px`
- **Tabs:** `0px` (Flat bottom border)

### Borders
- **Dark (Forensic Noir):** standard border `1px solid #1E1E20` (`UI_BORDER`). Active or hover: `1px solid #E8443A`. Verdict container top accent: `border-top: 3px solid #E8443A`.
- **Light (Forensic Lab):** default `1px solid #E2E5EA` (`UI_BORDER`); structure and tables may use `#AAB0B8` (`UI_STRUCTURE`) at reduced opacity. Active emphasis: `1px solid #C53030` (`ACCENT_TRACE`). Verdict strip: `border-top: 3px solid #D63031` (`ALERT_FORENSIC`).

---

## ✨ Gradients, Shadows & Overlays

### Background Texture
- **Noise Overlay**: A subtle fractal noise overlay (`4%` intensity, `0.35` opacity) gives a paper/analog texture.
- **Scanlines**: Repeating linear gradients (`rgba(0,0,0,0.03)`) at `4px` intervals simulate a CRT monitor.

### Gradients
EchoTrace uses precise gradients to direct user attention:
- **Hero Section:** `linear-gradient(180deg, #0D0D0E 0%, #0A0A0B 100%)`
- **Upload Zone (Top glow):** `radial-gradient(ellipse at 50% 0%, rgba(232, 68, 58, 0.04) 0%, transparent 70%)`
- **Mic Zone (Bottom glow):** `radial-gradient(ellipse at 50% 100%, rgba(232, 68, 58, 0.06) 0%, transparent 70%)`
- **Button Glass Effect:** `linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 50%)`
- **Progress Bar Fill:** `linear-gradient(90deg, #E8443A, #FF6B5E)`

### Box Shadows
Shadows are used sparingly, exclusively for highly interactive elements.
- **Button Hover Glow:** `0 8px 24px rgba(232, 68, 58, 0.35)`
- **Secondary Button Hover:** `0 4px 16px rgba(232, 68, 58, 0.2)`
- **Light surfaces:** prefer neutral elevation, not colored glow—e.g. `0 1px 3px rgba(15, 23, 42, 0.08)` on `BG_ELEVATED` cards; accent glow only on primary buttons at low alpha (`rgba(197, 48, 48, 0.25)`).

---

## 🎬 Micro-Animations
- `fadeInUp`: Used for revealing forensic cards smoothly (`0.8s cubic-bezier(0.16, 1, 0.3, 1)`).
- `pulse-dot`: A pulsing red dot animation for "LIVE" indicators and hero badges (`0.4` to `1` opacity).
- **Hover Transitions:** `0.2s ease` or `0.25s` for borders and button transform (`translateY(-1px)`).
