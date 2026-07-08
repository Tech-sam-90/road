# Colab runner

Code + data live on **Google account B**'s Drive. Compute runs in a Colab
runtime signed in under **Google account A**. `rclone` bridges the two: its
OAuth token is scoped to account B's Drive and stored independently of
whichever account's browser tab is running the notebook, so account A can
run indefinitely without ever re-authenticating through a Drive popup.

## One-time setup (do this yourself — it's an interactive OAuth step Claude
Code can't perform)

### 1. Get the repo onto Drive B

Under whatever Google Drive client/web UI account B uses, upload (or
`rclone copy` from your local machine once you've done step 2 below) the
whole `road-barbados-htr/` folder to the root of My Drive, so it ends up at
`My Drive/road-barbados-htr/...` — matching the layout this runner set
expects (`data/`, `src/`, `model_cache/`, `experiments/`, etc).

### 2. Authorize rclone against account B

On a machine with a browser (your laptop — not Colab):

```bash
rclone config
```

Walk through the prompts exactly like this:

1. `n` — New remote
2. `name>` — type **`roadB`** exactly. Every script in this runner set
   (`bootstrap.ipynb`, this README) hardcodes that name.
3. `Storage>` — type `drive` and pick "Google Drive" from the filtered list.
4. `client_id>` — leave blank (uses rclone's shared default client; fine for
   personal use).
5. `client_secret>` — leave blank.
6. `scope>` — choose `1` (`drive` — full access to all files). Scope `2`
   only sees files rclone itself created, which won't include the folder you
   uploaded in step 1.
7. `root_folder_id>` — leave blank.
8. `service_account_file>` — leave blank.
9. `Edit advanced config?` — `n`
10. `Use auto config?` — `y` **only if this machine has a browser rclone can
    open**. This is the critical step: when the browser opens,
    **make sure you're signed into (or switch to an incognito window signed
    into) Google account B** — not whatever account you normally browse
    with. The refresh token rclone stores is permanently scoped to whichever
    account approves this prompt.
    - If you're on a headless machine with no browser, use
      `rclone authorize "drive"` on a *different* machine that does have
      one, then paste the resulting config blob back when prompted.
11. Approve the requested Drive access in the browser, see "Success".
12. `Configure this as a Shared Drive (Team Drive)?` — `n`, unless
    `road-barbados-htr/` actually lives in a Shared Drive, in which case `y`
    and pick it from the list.
13. `y` — Yes this is OK
14. `q` — Quit config

This writes a `[roadB]` section into your local `rclone.conf`
(`~/.config/rclone/rclone.conf` on Linux/Mac, `%APPDATA%\rclone\rclone.conf`
on Windows) containing a refresh token bound to account B.

Sanity check before moving on:

```bash
rclone lsd roadB:
```

should list account B's Drive folders with no further prompts. If
`road-barbados-htr/` shows up there, you're set.

### 3. Put the config into a Colab secret (for account A)

1. Open the `rclone.conf` file from step 2 in a text editor, copy its
   **entire contents**.
2. In Colab (signed in as account A), open `bootstrap.ipynb`, click the key
   icon (🔑 "Secrets") in the left sidebar.
3. "+ Add new secret" → Name: **`RCLONE_CONF`** (must match exactly — the
   notebook reads this name), Value: paste the full file contents.
4. Toggle "Notebook access" on for this notebook.

Never paste the token into a code cell or let a cell print it — the
notebook's restore step writes it straight to
`~/.config/rclone/rclone.conf` on the Colab VM without echoing it, and
that file is never committed to git (Colab VMs are ephemeral anyway, but
don't rely on that).

## Remote naming convention

Everything in this runner set assumes:
- remote name: `roadB`
- repo root on Drive: `roadB:road-barbados-htr`
- data: `roadB:road-barbados-htr/data/{images,*.csv}`
- model cache: `roadB:road-barbados-htr/model_cache/`
- checkpoints: `roadB:road-barbados-htr/experiments/<tier>/`

If you rename anything, update `REMOTE` / `DRIVE_ROOT` at the top of
`bootstrap.ipynb` to match.

## Running a session

Open `bootstrap.ipynb` in Colab, run all cells top to bottom. It: checks
your GPU tier (T4/L4/A100/etc — not guaranteed, especially on free tier),
installs rclone, restores the secret, pulls code + data + cached model
weights + any existing checkpoint down to local Colab disk (`/content/...`
— always train against local disk, never the `roadB:` remote directly),
reinstalls `requirements.txt` fresh, and sets up checkpoint resume plus a
periodic push-back callback for `src/vlm/train.py` (built in a later
prompt) to use.

## Switching Colab accounts

GPU availability and usage limits are tied to account **A** (whichever
account is running the Colab runtime) — Drive storage and quota stay on
account **B** regardless. If account A gets rate-limited or runs out of
compute units:

1. Sign into Colab with a different Google account (call it A2).
2. Open `bootstrap.ipynb` there, add the same `RCLONE_CONF` secret (copy
   the same `rclone.conf` contents from step 3 above — it's still scoped to
   account B, unaffected by which account runs Colab).
3. Run all cells as normal.

None of the runner scripts depend on which account is running the
notebook — only on the `RCLONE_CONF` secret being present and the `roadB`
remote resolving correctly.
