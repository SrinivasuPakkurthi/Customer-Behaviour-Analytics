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


def validate_config(cfg: dict) -> list[str]:
    """
    Sanity-check an email config dict beyond just "are the fields filled in"
    (which is what config_is_complete checks). Returns a list of human
    readable warning strings; an empty list means the config looks fine.
    """
    warnings = []

    sender = cfg.get("sender_email", "")
    recipient = cfg.get("recipient_email", "")
    host = cfg.get("smtp_host", "")
    port = cfg.get("smtp_port", "")

    if sender and "@" not in sender:
        warnings.append(f"sender_email '{sender}' doesn't look like a valid email address")

    if recipient and "@" not in recipient:
        warnings.append(f"recipient_email '{recipient}' doesn't look like a valid email address")

    if not host:
        warnings.append("smtp_host is not set")

    try:
        port_int = int(port)
        if not (0 < port_int < 65536):
            warnings.append(f"smtp_port {port} is out of valid range (1-65535)")
    except (TypeError, ValueError):
        warnings.append(f"smtp_port '{port}' is not a valid number")

    if not cfg.get("sender_password"):
        warnings.append("sender_password is not set")

    return warnings


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


def send_generic_email(to_email: str, subject: str, plain_body: str,
                        html_body: Optional[str] = None,
                        cfg: Optional[dict] = None) -> tuple[bool, str]:
    """
    Send a plain (optionally HTML) email using the admin-configured SMTP
    settings (the same ones used for at-risk alerts). Used for things like
    password-reset messages that aren't tied to a specific customer record.

    Returns (success: bool, message: str).
    """
    if cfg is None:
        cfg = load_email_config()

    if not cfg.get("sender_email") or not cfg.get("sender_password"):
        return False, "Email sending is not configured yet (no sender configured in Email Alerts)."

    warnings = validate_config(cfg)
    if warnings:
        return False, "Email config looks off: " + " | ".join(warnings)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = cfg["sender_email"]
    msg["To"] = to_email

    msg.attach(MIMEText(plain_body, "plain"))
    if html_body:
        msg.attach(MIMEText(html_body, "html"))

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"])) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(cfg["sender_email"], cfg["sender_password"])
            server.sendmail(cfg["sender_email"], to_email, msg.as_string())
        logger.info("Generic email sent to %s", to_email)
        return True, f"Message sent to {to_email}."
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Check email/password (Gmail: use an App Password)."
    except Exception as exc:
        logger.error("Failed to send generic email to %s: %s", to_email, exc)
        return False, f"Failed to send: {exc}"


def send_password_reset_email(username: str, to_email: str, new_password: str,
                               cfg: Optional[dict] = None) -> tuple[bool, str]:
    """
    Send a password-reset message containing a freshly generated temporary
    password for `username`. Returns (success: bool, message: str).
    """
    subject = "🔑 Your password has been reset"
    plain = (
        f"Hi {username},\n\n"
        f"You requested a password reset for your Customer Behaviour & Churn "
        f"Analytics account.\n\n"
        f"Your new temporary password is: {new_password}\n\n"
        f"Please sign in with this password and change it right away from "
        f"the 'Change Password' page.\n\n"
        f"If you did not request this, please contact your administrator."
    )
    html = f"""
    <html><body style="font-family:Arial,sans-serif;background:#f7f9fc;padding:24px;">
      <div style="max-width:520px;margin:auto;background:#fff;border-radius:12px;
                  box-shadow:0 2px 12px rgba(0,0,0,0.08);overflow:hidden;">
        <div style="background:#4F46E5;padding:22px 28px;">
          <h1 style="color:#fff;margin:0;font-size:20px;">🔑 Password Reset</h1>
        </div>
        <div style="padding:26px 28px;">
          <p style="color:#374151;font-size:15px;">Hi <strong>{username}</strong>,</p>
          <p style="color:#374151;font-size:15px;">
            Your new temporary password is:
          </p>
          <p style="font-size:20px;font-weight:700;letter-spacing:1px;
                     background:#f3f4f6;padding:12px 16px;border-radius:8px;
                     color:#111827;">{new_password}</p>
          <p style="color:#6b7280;font-size:13.5px;">
            Please sign in and change your password immediately from the
            <strong>Change Password</strong> page. If you did not request
            this reset, contact your administrator.
          </p>
        </div>
      </div>
    </body></html>
    """
    return send_generic_email(to_email, subject, plain, html, cfg)


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
