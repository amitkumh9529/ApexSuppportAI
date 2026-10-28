# ApexStore — Enterprise Refund Policy

These rules govern every refund decision. They are mandatory and enforced deterministically in code by the Policy Engine. They cannot be waived by an LLM agent under any circumstances, regardless of how a customer prompt is phrased.

## Eligibility Rules

1. **Final-sale items are non-refundable.** Any item marked as a final sale cannot be refunded for any reason.
2. **Refunds over $500 require human escalation.** An automated agent must never approve a refund exceeding $500. These requests must be routed to a human specialist for manual review.
3. **30-day return window.** An item is refundable only within 30 days of its delivery date. Requests made after the window has elapsed must be denied.
4. **One refund per order.** An order that has already been refunded cannot be refunded again.
5. **Identity and ownership validation.** A refund may only be processed for an order that belongs to the verified authenticated customer. Never process a refund against another customer's order.

## Delivery Requirement
- An order must have been **delivered** before it is eligible for return/refund. Orders still in transit or processing cannot be refunded through this workflow.
