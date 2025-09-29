class ListNode():
    def __init__(self,val,next=None):
        self.val=val
        self.next=next

def reverseKGroupRecursive(head,k):
    curr=head
    count=0
    while curr and count<k:
        curr=curr.next
        count+=1

    if count==k:
        new_head=reverse(head,k)
        head.next=reverseKGroupRecursive(curr,k)
        return new_head
    return head

def reverse(head,k):
    prev=None
    curr=head
    for _ in range(k):
        next_node=curr.next
        curr.next=prev
        prev=curr
        curr=next_node
    return prev 

def print_list(head):
    while head:
        print(head.val)
        head=head.next

head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
new_head_rec = reverseKGroupRecursive(head, 2)
print("递归法结果: ", end="")
print_list(new_head_rec)
