"""
Email Alert System for Customer Churn Risk.

Sends email notifications when a customer's archetype transitions
to 'at_risk'. Uses Gmail SMTP by default (works with any SMTP server).

Features:
- Single customer alert (triggered automatically on archetype change)
- Bulk alert for all current at-risk customers
- Alert history log saved to data/alert_log.csv
- Config stored in data/email_config.json
"""

import json
import logging
import os
import smtplib
import ssl
import time
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CONFIG_PATH   = "data/email_config.json"
ALERT_LOG_PATH = "data/alert_log.csv"

DEFAULT_CONFIG = {
    "smtp_host":     "smtp.gmail.com",
    "smtp_port":     587,
    "sender_email":  "",
    "sender_password": "",   # Gmail: use an App Password, not your main password
    "recipient_email": "",
    "alerts_enabled": False,
}


# ----------------------------------------------------------------------------
# Config helpers
# ----------------------------------------------------------------------------

def load_email_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                cfg = json.load(f)
            # fill any missing keys from defaults
            for k, v in DEFAULT_CONFIG.items():
                cfg.setdefault(k, v)
            return cfg
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_email_config(cfg: dict) -> None:
    os.makedirs("data", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Email config saved")


def config_is_complete(cfg: dict) -> bool:
    return all([
        cfg.get("sender_email"),
        cfg.get("sender_password"),
        cfg.get("recipient_email"),
        cfg.get("smtp_host"),
        cfg.get("smtp_port"),
        cfg.get("alerts_enabled", False),
    ])


# ----------------------------------------------------------------------------
# Alert log helpers
# ----------------------------------------------------------------------------

def load_alert_log() -> pd.DataFrame:
    if os.path.exists(ALERT_LOG_PATH):
        try:
            return pd.read_csv(ALERT_LOG_PATH, parse_dates=["sent_at"])
        except Exception:
            pass
    return pd.DataFrame(columns=["sent_at", "customer_id", "name",
                                   "archetype", "status", "error"])


def _append_alert_log(customer_id: str, name: str, archetype: str,
                       status: str, error: str = "") -> None:
    log = load_alert_log()
    new_row = pd.DataFrame([{
        "sent_at":     datetime.now(),
        "customer_id": customer_id,
        "name":        name,
        "archetype":   archetype,
        "status":      status,
        "error":       error,
    }])
    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(ALERT_LOG_PATH, index=False)


# ----------------------------------------------------------------------------
# Email builder
# ----------------------------------------------------------------------------

def _build_email_html(customer: dict, cfg: dict) -> str:
    archetype   = customer.get("archetype", "at_risk")
    name        = customer.get("name", "Unknown")
    cid         = customer.get("customer_id", "")
    recency     = int(customer.get("recency_days", 0))
    frequency   = int(customer.get("frequency", 0))
    monetary    = float(customer.get("monetary", 0))
    segment     = customer.get("RFM_segment", "N/A")
    clv         = float(customer.get("CLV", 0))

    color = "#dc2626" if archetype == "at_risk" else "#f59e0b"
    label = archetype.replace("_", " ").title()

    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#f7f9fc;padding:24px;">
      <div style="max-width:600px;margin:auto;background:#fff;border-radius:12px;
                  box-shadow:0 2px 12px rgba(0,0,0,0.08);overflow:hidden;">

        <div style="background:{color};padding:24px 32px;">
          <h1 style="color:#fff;margin:0;font-size:22px;">
            ⚠️ Customer At-Risk Alert
          </h1>
          <p style="color:#fff;margin:6px 0 0;opacity:0.9;">
            Customer Behaviour Analysis System
          </p>
        </div>

        <div style="padding:28px 32px;">
          <p style="color:#374151;font-size:15px;">
            A customer has been flagged as <strong style="color:{color};">{label}</strong>
            and may require immediate attention.
          </p>

          <table style="width:100%;border-collapse:collapse;margin:20px 0;">
            <tr style="background:#f3f4f6;">
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;width:40%;">Customer ID</td>
              <td style="padding:10px 14px;color:#111827;">{cid}</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Name</td>
              <td style="padding:10px 14px;color:#111827;">{name}</td>
            </tr>
            <tr style="background:#f3f4f6;">
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Archetype</td>
              <td style="padding:10px 14px;">
                <span style="background:{color}22;color:{color};padding:3px 10px;
                             border-radius:20px;font-weight:600;">{label}</span>
              </td>
            </tr>
            <tr>
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Days Since Last Purchase</td>
              <td style="padding:10px 14px;color:#111827;">{recency} days</td>
            </tr>
            <tr style="background:#f3f4f6;">
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Total Purchases</td>
              <td style="padding:10px 14px;color:#111827;">{frequency}</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Total Spending</td>
              <td style="padding:10px 14px;color:#111827;">₹{monetary:,.2f}</td>
            </tr>
            <tr style="background:#f3f4f6;">
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">RFM Segment</td>
              <td style="padding:10px 14px;color:#111827;">{segment}</td>
            </tr>
            <tr>
              <td style="padding:10px 14px;font-weight:600;color:#6b7280;">Lifetime Value (CLV)</td>
              <td style="padding:10px 14px;color:#111827;">₹{clv:,.2f}</td>
            </tr>
          </table>

          <div style="background:#fef2f2;border-left:4px solid {color};
                      padding:14px 18px;border-radius:6px;margin:16px 0;">
            <p style="margin:0;font-weight:600;color:{color};">Recommended Actions</p>
            <ul style="margin:8px 0 0;color:#374151;padding-left:18px;">
              <li>Launch a win-back campaign with a special discount</li>
              <li>Send a personalised re-engagement email or SMS</li>
              <li>Offer loyalty point bonuses for the next purchase</li>
              <li>Schedule a customer service check-in call</li>
            </ul>
          </div>

          <p style="color:#9ca3af;font-size:13px;margin-top:24px;">
            Sent by Customer Behaviour Analysis & Churn Prediction System
            · {datetime.now().strftime("%d %b %Y, %H:%M")}
          </p>
        </div>
      </div>
    </body></html>
    """
    return html


# ----------------------------------------------------------------------------
# Core send function
# ----------------------------------------------------------------------------

def send_alert_email(customer: dict, cfg: Optional[dict] = None) -> tuple[bool, str]:
    """
    Send an at-risk alert email for a single customer.

    Returns (success: bool, message: str).
    """
    if cfg is None:
        cfg = load_email_config()

    if not config_is_complete(cfg):
        return False, "Email alerts are not configured or disabled."

    warnings = validate_config(cfg)
    if warnings:
        err = "Email config looks off before we even tried to send: " + " | ".join(warnings)
        _append_alert_log(
            customer.get("customer_id", ""),
            customer.get("name", "Unknown"),
            customer.get("archetype", "at_risk"),
            "failed",
            err,
        )
        return False, err

    name   = customer.get("name", "Unknown")
    cid    = customer.get("customer_id", "")
    label  = customer.get("archetype", "at_risk").replace("_", " ").title()

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"⚠️ At-Risk Customer Alert: {name} ({cid})"
    msg["From"]    = cfg["sender_email"]
    msg["To"]      = cfg["recipient_email"]

    plain = (
        f"At-Risk Customer Alert\n\n"
        f"Customer: {name} ({cid})\n"
        f"Archetype: {label}\n"
        f"Days since last purchase: {int(customer.get('recency_days', 0))}\n"
        f"Total spending: ₹{float(customer.get('monetary', 0)):,.2f}\n\n"
        f"Recommended: Launch a win-back campaign immediately."
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(_build_email_html(customer, cfg), "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.sendmail(cfg["sender_email"], cfg["recipient_email"], msg.as_string())
        _append_alert_log(cid, name, customer.get("archetype", "at_risk"), "sent")
        logger.info("Alert email sent for customer %s", cid)
        return True, f"✅ Alert email sent for {name} ({cid})"
    except smtplib.SMTPAuthenticationError:
        err = "Authentication failed. Check email/password (Gmail: use an App Password)."
        _append_alert_log(cid, name, customer.get("archetype", "at_risk"), "failed", err)
        return False, err
    except Exception as exc:
        err = str(exc)
        _append_alert_log(cid, name, customer.get("archetype", "at_risk"), "failed", err)
        logger.error("Failed to send alert for %s: %s", cid, err)
        return False, f"Failed to send: {err}"


def send_bulk_alerts(features: pd.DataFrame,
                      cfg: Optional[dict] = None,
                      limit: Optional[int] = None,
                      delay_seconds: float = 1.5,
                      progress_callback=None) -> tuple[int, int, list]:
    """
    Send alerts for customers currently classified as 'at_risk'.

    Args:
        features: feature table containing at-risk customers.
        cfg: email config dict (loaded from disk if not provided).
        limit: if set, only send to the first N at-risk customers
               (useful for testing before doing a full send).
        delay_seconds: pause between each send to avoid tripping
               Gmail's spam/rate-limit detection on bulk sends.
        progress_callback: optional callable(sent, failed, total) invoked
               after each send, e.g. to update a UI progress bar.

    Returns (sent_count, failed_count, messages).
    """
    if cfg is None:
        cfg = load_email_config()

    at_risk = features[features["archetype"] == "at_risk"]
    if limit is not None:
        at_risk = at_risk.head(limit)

    total = len(at_risk)
    sent, failed, messages = 0, 0, []

    for i, (_, row) in enumerate(at_risk.iterrows()):
        ok, msg = send_alert_email(row.to_dict(), cfg)
        messages.append(msg)
        if ok:
            sent += 1
        else:
            failed += 1

        if progress_callback is not None:
            progress_callback(sent, failed, total)

        # Pause between sends to avoid Gmail rate-limiting / spam flags,
        # but skip the delay after the very last email.
        if delay_seconds > 0 and i < total - 1:
            time.sleep(delay_seconds)

    return sent, failed, messages


def check_and_alert_new_at_risk(old_features: pd.DataFrame,
                                  new_features: pd.DataFrame,
                                  cfg: Optional[dict] = None) -> list[str]:
    """
    Compare old vs new feature tables to find customers who have JUST
    transitioned into the 'at_risk' archetype, and send alerts for them.

    Returns list of result messages.
    """
    if cfg is None:
        cfg = load_email_config()
    if not config_is_complete(cfg):
        return []

    old_map = old_features.set_index("customer_id")["archetype"].to_dict()
    messages = []

    for _, row in new_features.iterrows():
        cid = row["customer_id"]
        new_arch = row["archetype"]
        old_arch = old_map.get(cid, "new")   # brand-new customers default to "new"

        if new_arch == "at_risk" and old_arch != "at_risk":
            ok, msg = send_alert_email(row.to_dict(), cfg)
            messages.append(msg)

    return messages
